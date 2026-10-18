"""CPU contract regressions. Real source definitions, mocked CUDA/I/O seams only.

Run: .venv-tp1-cpu/bin/python -m unittest discover -s tests/cpu -p test_glm_tp1_contracts.py -v
No torch, CUDA import/build, checkpoint load or serving measurement is implied.
"""
from __future__ import annotations

import ast
from contextlib import nullcontext
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "src/tensorfold/families/glm5_next"
SCRATCH = Path(os.environ.get("TMPDIR", "/opt/data/cache/scratch"))
SCRATCH.mkdir(parents=True, exist_ok=True)


def source_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def definitions(path, names, env=None):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    scope = {"__name__": "cpu_probe", **(env or {})}
    exec(compile(ast.Module(nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope


split = source_module(BASE / "cuda/split.py", "split_probe")
engine = source_module(BASE / "cuda/engine.py", "tensorfold.families.glm5_next.cuda.engine")
family = source_module(BASE / "__init__.py", "tensorfold.families.glm5_next")


class ConstructorHarness:
    """Only hardware/import boundaries replaced; production constructor runs intact."""
    def __init__(self, root, world=1, mtp=True):
        self.calls = []
        self.cfg = NS(dense_limit=2051, eos=(7,), quant="mlx", layers=45, mtp_layers=1)
        self.w = NS(cfg=self.cfg, mtp=object() if mtp else None, comm=None, world=world)
        self.text = {"num_hidden_layers": 45, "num_nextn_predict_layers": 1, "_quantization": {}}
        self.modules = {}
        for name in ("torch", "tensorfold", "tensorfold.cuda", "tensorfold.families",
                     "tensorfold.families.glm5_next", "tensorfold.families.glm5_next.cuda"):
            self.modules[name] = ModuleType(name)
        torch = self.modules["torch"]
        torch.cuda = NS(set_device=lambda n: self.calls.append(("device", n)))
        self.modules["tensorfold.families.glm5_next.cuda"].LATENT = True
        self.comm = NS(barrier=lambda: self.calls.append(("barrier",)), all_gather=lambda *_: None,
                       ready=lambda *_, **__: None)
        def nccl(rank, world, master, port):
            self.calls.append(("nccl", rank, world)); return self.comm
        self.add("tensorfold.cuda.comm", NCCL=nccl, open_comm=nccl)
        def load(path, **kwargs):
            self.calls.append(("load", kwargs))
            if not kwargs.get("load_mtp", True):
                self.w.mtp = None
            return self.w
        self.add("tensorfold.families.glm5_next.cuda.weights", Config=NS(read=lambda _: self.cfg), load=load)
        def decoder(w, **kwargs):
            self.calls.append(("decoder", kwargs)); return NS(st=NS(), reset=lambda: None)
        self.add("tensorfold.families.glm5_next.cuda.decode", Engine=decoder)
        self.modules["tensorfold.families.glm5_next.cuda.split"] = split
        geom_path = ROOT / "src/tensorfold/cuda/geometry.py"
        scope = definitions(geom_path, {"size", "padded", "split_weights"}, {"math": math, "itemsize": lambda i, n: split.DTYPE_BYTES[i["dtype"]]})
        def geometry(text, w, reserve, **kwargs):
            self.calls.append(("geometry", w, text.get("num_nextn_predict_layers"), kwargs)); return NS()
        self.add("tensorfold.cuda.geometry", PREFILL_ROWS=2048, mla_geometry=geometry,
                 draft_geometry=geometry, dflash2_geometry=geometry, dflash2_weights=lambda d, w: NS(),
                 split_weights=scope["split_weights"])
        def admit(path, requested, explicit, torch, geometry, transform, **kwargs):
            self.calls.append(("admit", kwargs))
            geometry(self.text)
            names = {
                "model.language_model.layers.0.self_attn.q_proj.weight": {"shape": [8, 4], "dtype": "F16"},
                "model.language_model.layers.45.eh_proj.trellis": {"shape": [4, 4, 32], "dtype": "I16"},
                "lm_head.trellis": {"shape": [8, 4, 80], "dtype": "I16"},
                "model.visual.blocks.0.weight": {"shape": [8, 4], "dtype": "F16"},
            }
            self.estimates = {n: transform(n, i)[0] for n, i in names.items()}
            return {"context_window": requested or 2051, "cache_slots": 2560, "budget_bytes": 1 << 36,
                    "total_bytes_estimate": 1 << 30, "serving_peak_bytes_estimate": 1 << 30}
        self.add("tensorfold.cuda.capacity", admit=admit)

    def add(self, name, **attrs):
        self.modules[name] = ModuleType(name)
        self.modules[name].__dict__.update(attrs)

    def construct(self, path, **kwargs):
        def gather(e, values):
            return [list(values) for _ in range(e.world)]
        with patch.dict(sys.modules, self.modules), patch.object(engine.GlmEngine, "_gather_ints", gather), \
             patch.object(engine.GlmEngine, "_calibrate", lambda e: self.calls.append(("calibrate",)) or {"timed": {}, "verify": [1] * 8, "mtp": 0, "mtp_step": 0, "mtp_row": 0, "block": 0, "taps_row": 0}):
            return engine.GlmEngine(path, rank=0, master="test", port=1, **kwargs)


class SplitContracts(unittest.TestCase):
    def test_world1_bytes_all_partition_kinds_identity(self):
        for shape, kind in (([3, 5], "row"), ([3, 5], "col"), ([3, 5, 7], "dim1"), ([3], "rep")):
            raw = np.arange(math.prod(shape) * 2, dtype=np.uint8)
            data, out_shape = split.split_bytes(raw, shape, 2, kind, 0, world=1)
            self.assertIs(data, raw)
            self.assertEqual(out_shape, shape)

    def test_world2_legacy_bytes_exact(self):
        for kind, shape, axis in (("row", [8, 6], 0), ("col", [8, 6], 1), ("dim1", [8, 6, 4], 1)):
            raw = np.arange(math.prod(shape) * 2, dtype=np.uint8)
            parts = [split.split_bytes(raw, shape, 2, kind, r) for r in (0, 1)]
            shaped = [a.reshape(*s, 2) for a, s in parts]
            np.testing.assert_array_equal(np.concatenate(shaped, axis=axis).reshape(-1), raw)

    def test_invalid_world_rank_fails(self):
        for world, rank in ((1, 1), (2, 2), (0, 0), (3, 0), (2, -1)):
            with self.assertRaises(ValueError):
                split.split_bytes(np.arange(4, dtype=np.uint8), [4], 1, "rep", rank, world=world)

    def test_world1_device_clone_unsplit(self):
        class Raw:
            def clone(self): return "owned"
        data, shape = split.split_device(Raw(), [3, 5], 2, "col", 0, world=1)
        self.assertEqual((data, shape), ("owned", [3, 5]))

    def test_world1_span_full_fused_and_packed(self):
        reader = split.RankReader.__new__(split.RankReader)
        reader.dir, reader.rank, reader.world, reader.split = Path("unused"), 0, 1, False
        names = ["model.language_model.layers.0.self_attn.qkv_proj.trellis",
                 "model.language_model.layers.0.self_attn.conv1d.weight",
                 "model.language_model.layers.3.mlp.experts.0.gate_proj.mul1"]
        reader.index = dict.fromkeys(names, "shard")
        reader.files = {"unused/shard": ({n: {"data_offsets": [0, 30], "shape": [3, 5], "dtype": "F16"} for n in names}, 16)}
        for n in names:
            self.assertEqual(reader._span(n), ("unused/shard", 16, 46, "rep", [3, 5], "F16"))

    def test_world2_span_still_reads_half_rows(self):
        reader = split.RankReader.__new__(split.RankReader)
        reader.dir, reader.rank, reader.world, reader.split = Path("unused"), 1, 2, False
        n = "model.language_model.layers.0.self_attn.q_proj.weight"
        reader.index = {n: "shard"}
        reader.files = {"unused/shard": ({n: {"data_offsets": [0, 64], "shape": [8, 4], "dtype": "F16"}}, 16)}
        self.assertEqual(reader._span(n), ("unused/shard", 48, 80, "rep", [4, 4], "F16"))

    def test_singleton_rejects_index_referencing_rank_shards(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as folder:
            Path(folder, "config.json").write_text("{}")
            Path(folder, "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"lm_head.trellis": "../rank0/model.rank0.safetensors"}}))
            with self.assertRaisesRegex(ValueError, "full|unsplit|rank"):
                split.require_full_checkpoint(folder, 1)

    def test_singleton_rejects_half_rank_folder_before_io(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as folder:
            Path(folder, "model.rank0.safetensors").touch()
            module = ModuleType("tensorfold.cuda.direct_read")
            module.Reader = lambda: self.fail("half-folder rejection must precede I/O construction")
            module.ReadAhead = lambda *_: None
            module.SafeTensors = lambda *_: None
            with patch.dict(sys.modules, {"tensorfold.cuda.direct_read": module}):
                with self.assertRaisesRegex(ValueError, "full|unsplit|rank"):
                    split.RankReader(folder, 0, world=1)


class EngineContracts(unittest.TestCase):
    def test_singleton_local_transport_without_torch(self):
        e = engine.GlmEngine.__new__(engine.GlmEngine)
        e.world, e.rank, e.comm = 1, 0, None
        self.assertEqual(e._gather_ints([4, 9]), [[4, 9]])
        self.assertEqual(e._share([4, 9]), [4, 9])
        self.assertEqual(e._share([]), [])
        with self.assertRaises(ValueError): e._share(None)

    def test_singleton_constructor_target_only_eager(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as folder:
            h = ConstructorHarness(folder)
            e = h.construct(Path(folder), world=1, target_only=True, eager=True)
        self.assertIsNone(e.comm)
        self.assertFalse(any(c[0] in ("nccl", "barrier", "calibrate") for c in h.calls))
        load = next(c[1] for c in h.calls if c[0] == "load")
        self.assertEqual((load["world"], load["load_mtp"]), (1, False))
        decode = next(c[1] for c in h.calls if c[0] == "decoder")
        self.assertFalse(decode["graphs"])
        self.assertEqual(next(c[1]["world"] for c in h.calls if c[0] == "admit"), 1)
        self.assertEqual(next(c[1:3] for c in h.calls if c[0] == "geometry"), (1, 0))
        self.assertEqual(h.estimates["model.language_model.layers.0.self_attn.q_proj.weight"], 64)
        self.assertEqual(h.estimates["lm_head.trellis"], 5120)
        self.assertEqual(h.estimates["model.language_model.layers.45.eh_proj.trellis"], 0)
        self.assertEqual(h.estimates["model.visual.blocks.0.weight"], 0)
        self.assertEqual(e.cache_bytes, 0)
        self.assertEqual(e.policy, "0")

    def test_tp2_constructor_preserves_communication_and_mtp(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as folder:
            h = ConstructorHarness(folder, world=2)
            e = h.construct(Path(folder), world=2)
        self.assertIs(e.comm, h.comm)
        self.assertIn(("nccl", 0, 2), h.calls)
        self.assertEqual(sum(c[0] == "barrier" for c in h.calls), 2)
        self.assertIn(("calibrate",), h.calls)
        self.assertEqual(next(c[1]["world"] for c in h.calls if c[0] == "load"), 2)
        self.assertEqual(h.estimates["model.language_model.layers.0.self_attn.q_proj.weight"], 32)

    def test_packed_admission_uses_adapter_residency_and_workspace(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as folder:
            h = ConstructorHarness(folder)
            h.cfg.quant = "exl3"
            cap = h.modules["tensorfold.cuda.capacity"]
            cap.headers = lambda path, **kwargs: {"lm_head.trellis": {"shape": [8, 4, 80], "dtype": "I16"}}
            weights = h.modules["tensorfold.families.glm5_next.cuda.weights"]
            def resident(cfg, *, load_mtp):
                h.calls.append(("packed_residency", load_mtp))
                return lambda n, i: (123 if "visual" not in n and ".45." not in n else 0, 0)
            weights.packed_residency = resident
            def workspace(cfg, headers, **kwargs):
                h.calls.append(("workspace", kwargs)); return 4096
            weights.workspace_bytes = workspace
            h.modules["tensorfold.cuda.geometry"].with_fixed = lambda geo, n: h.calls.append(("extra", n)) or geo
            e = h.construct(Path(folder), world=1, target_only=True, eager=True)
        self.assertIn(("packed_residency", False), h.calls)
        self.assertIn(("extra", 4096), h.calls)
        self.assertEqual(h.estimates["lm_head.trellis"], 123)
        self.assertEqual(next(c[1]["load_mtp"] for c in h.calls if c[0] == "workspace"), False)

    def test_serial_only_disables_mtp_startup_and_calibration(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as folder:
            h = ConstructorHarness(folder)
            e = h.construct(Path(folder), world=1, serial_only=True)
        self.assertIsNone(e.w.mtp)
        self.assertFalse(any(c[0] == "calibrate" for c in h.calls))
        self.assertEqual(e._drafters([0, 0, 0, 0]), (False, False, False))

    def test_eager_skips_calibration(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as folder:
            h = ConstructorHarness(folder)
            h.construct(Path(folder), world=1, eager=True, policy="1")
        self.assertFalse(any(c[0] == "calibrate" for c in h.calls))
        self.assertFalse(next(c[1]["graphs"] for c in h.calls if c[0] == "decoder"))

    def test_eager_rejects_auto_policy_without_timing(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as folder:
            h = ConstructorHarness(folder)
            e = h.construct(Path(folder), world=1, eager=True)
        with self.assertRaisesRegex(ValueError, "calibrat|automatic|fixed"):
            e._effective(engine.encode_policy("auto"))

    def test_target_only_never_selects_absorption_or_drafts(self):
        e = engine.GlmEngine.__new__(engine.GlmEngine)
        e.target_only, e.load_mtp, e.drafter, e.serial_only = True, False, None, True
        e.w = NS(mtp=None)
        for spec in ("0", "1", "auto", "fc5:0.3"):
            code = e._effective(engine.encode_policy(spec))
            self.assertEqual(code, [0, 0, 0, 0])
            self.assertEqual(e._drafters(code), (False, False, False))

    def test_target_only_request_executes_serial_without_absorption_or_snapshot(self):
        e = engine.GlmEngine.__new__(engine.GlmEngine)
        e.target_only, e.serial_only, e.load_mtp, e.calibrated = True, True, False, False
        e.drafter, e.w, e.eos = None, NS(mtp=None), (7,)
        e.cache_entries, e.cache_bytes, e.cache, e.live = 0, 0, [], []
        e.e = NS(st=NS(pos=2), last_hidden=None)
        observed = []
        def prefill(e, prompt, sampling, **kwargs):
            observed.append(("prefill", kwargs)); return 11
        def serial(e, first, max_tokens, sampling, **kwargs):
            observed.append(("serial", first, max_tokens))
            return NS(tokens=[11, 12], seconds=0, rounds=1, depths=[], arms=None, stages=None)
        def forbidden(*args, **kwargs): self.fail("draft/snapshot path reached")
        decode = ModuleType("tensorfold.families.glm5_next.cuda.decode")
        decode.__dict__.update(DepthPolicy=forbidden, dflash_decode=forbidden, mtp_decode=forbidden,
                              prefill=prefill, serial_decode=serial, take_snapshot=forbidden, load_rows=forbidden)
        choice = ModuleType("tensorfold.families.glm5_next.cuda.drafter_choice")
        choice.__dict__.update(DrafterChoice=forbidden, auto_decode=forbidden)
        e._take_over = lambda _: None
        with patch.dict(sys.modules, {decode.__name__: decode, choice.__name__: choice}):
            e._run_once([1, 2], 2, None, True, lambda ids: observed.append(("output", ids)),
                        engine.encode_policy("auto"), None, True)
        self.assertFalse(observed[0][1]["mtp"])
        self.assertIsNone(observed[0][1]["drafter"])
        self.assertIn(("serial", 11, 2), observed)

    def test_singleton_cannot_follow(self):
        e = engine.GlmEngine.__new__(engine.GlmEngine)
        e.world, e.rank = 1, 0
        with self.assertRaisesRegex(ValueError, "rank|TP|world"):
            e.follow()

    def test_zero_cache_does_not_keep_one_snapshot(self):
        e = engine.GlmEngine.__new__(engine.GlmEngine)
        e.cache, e.cache_entries, e.cache_bytes = [], 0, 0
        e._remember(NS(ids=[1]))
        self.assertEqual(e.cache, [])


class FamilyContracts(unittest.TestCase):
    def test_family_accepts_exact_mul1_config_and_preserves_other_gates(self):
        sys.path.insert(0, str(ROOT / "src"))
        from tensorfold import families
        cases = [({"quant_method": "exl3", "bits": 2.05, "head_bits": 5, "codebook": "mul1", "out_scales": "always"}, True),
                 ({"quant_method": "exl3", "bits": 4, "codebook": "mcg", "scope": "glm53_routed_experts_only"}, True),
                 ({"quant_method": "exl3", "bits": 3, "codebook": "mcg", "scope": "glm53_routed_experts_only"}, False),
                 ({"quant_method": "exl3", "bits": 2.05, "head_bits": 5, "codebook": "3inst", "out_scales": "always"}, False)]
        with tempfile.TemporaryDirectory(dir=SCRATCH) as folder:
            for quant, okay in cases:
                Path(folder, "config.json").write_text(json.dumps({"model_type": "glm5_next", "quantization_config": quant}))
                if okay:
                    family.check(folder)
                else:
                    with self.assertRaises(ValueError): family.check(folder)

    def test_family_forwards_world_and_modes(self):
        module = ModuleType("tensorfold.families.glm5_next.cuda.engine")
        module.DEFAULT_POLICY, module.DFLASH_POLICY = "auto", "fc5:0.3"
        module.GlmEngine = lambda path, **kwargs: kwargs
        with patch.dict(sys.modules, {module.__name__: module}):
            options = family.cuda_engine("unused", tp=1, target_only=True, eager=True)
        self.assertEqual(options["world"], 1)
        self.assertTrue(options["target_only"] and options["eager"])
        self.assertTrue(options["serial_only"])
        self.assertIsNone(options["drafter"])

    def test_invalid_family_ranks_and_tp2_master(self):
        for tp, rank, master in ((1, 1, ""), (2, 2, "ok"), (3, 0, "ok"), (2, 0, "")):
            with self.assertRaises(ValueError):
                family.cuda_engine("unused", tp=tp, rank=rank, master=master)


def _cli_args():
    """The parser module ``cli.build_parser`` delegates to since 0.6."""
    sys.path.insert(0, str(ROOT / "src"))
    from tensorfold import cli_args
    return cli_args


class CliContracts(unittest.TestCase):
    def test_parser_exposes_explicit_modes_without_changing_defaults(self):
        import argparse
        scope = definitions(ROOT / "src/tensorfold/cli.py", {"build_parser"},
                            {"argparse": argparse, "Path": Path, "__version__": "probe", "MEMORY_FRACTION": 0.8,
                             "cli_args": _cli_args(),
                             **{name: lambda: None for name in ("cmd_serve", "cmd_pull", "cmd_models", "cmd_info", "cmd_update", "cmd_plan")}})
        parser = scope["build_parser"]()
        args = parser.parse_args(["serve", "unused", "--target-only", "--eager"])
        self.assertTrue(args.target_only and args.eager)
        args = parser.parse_args(["serve", "unused"])
        self.assertFalse(args.target_only or args.eager)

    def test_cli_forwards_modes_and_never_resolves_auto_drafter(self):
        import argparse
        calls = []
        class ReachedEngine(Exception): pass
        def construct(path, **kwargs):
            calls.append(kwargs)
            raise ReachedEngine()
        env = {"argparse": argparse, "Path": Path, "__version__": "probe", "MEMORY_FRACTION": 0.8,
               "time": engine.time, "_vision_options": lambda args: {},
               "_drafter": lambda *args: self.fail("auto drafter resolved in target-only"),
               "cli_args": _cli_args(),
                             **{name: lambda: None for name in ("cmd_serve", "cmd_pull", "cmd_models", "cmd_info", "cmd_update", "cmd_plan")}}
        scope = definitions(ROOT / "src/tensorfold/cli.py", {"build_parser", "_serve_cuda", "_glm_modes"}, env)
        args = scope["build_parser"]().parse_args(["serve", "unused", "--target-only", "--eager"])
        sys.path.insert(0, str(ROOT / "src"))
        from tensorfold import hub
        with self.assertRaises(ReachedEngine):
            scope["_serve_cuda"](args, NS(model_type="glm5_next", title="GLM", package=NS(cuda_engine=construct)), Path("unused"))
        self.assertTrue(calls[0]["target_only"] and calls[0]["eager"] and calls[0]["no_drafts"])
        self.assertEqual(calls[0]["drafter"], "")

    def test_flags_reject_unsupported_family_or_backend(self):
        scope = definitions(ROOT / "src/tensorfold/cli.py", {"_glm_modes"})
        modes = scope["_glm_modes"]
        args = NS(target_only=True, eager=True)
        self.assertEqual(modes(args, NS(model_type="glm5_next"), "cuda"), {"target_only": True, "eager": True})
        for family_kind, backend in (("other", "cuda"), ("glm5_next", "mlx")):
            with self.assertRaises(ValueError): modes(args, NS(model_type=family_kind), backend)
        self.assertEqual(modes(NS(target_only=False, eager=False), NS(model_type="other"), "cuda"), {})


class TorchCpuContracts(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ImportError:
            self.skipTest("real tensor/reader CPU regressions need torch; run pinned CPU container")
        self.torch = torch

    def test_device_split_on_cpu_tensors_exact_and_owned(self):
        torch = self.torch
        for world in (1, 2):
            for kind, shape in (("row", [8, 6]), ("col", [8, 6]), ("dim1", [8, 6, 4]), ("rep", [8, 6])):
                raw = np.arange(math.prod(shape) * 2, dtype=np.uint8)
                for rank in range(world):
                    data, dims = split.split_device(torch.from_numpy(raw), shape, 2, kind, rank, world=world)
                    oracle, oracle_dims = split.split_bytes(raw, shape, 2, kind, rank, world=world)
                    self.assertEqual(dims, oracle_dims)
                    np.testing.assert_array_equal(data.numpy(), oracle)
                    self.assertNotEqual(data.data_ptr(), torch.from_numpy(raw).data_ptr())

    def test_tp2_transport_real_cpu_tensors(self):
        torch = self.torch
        def cpu(fn):
            def call(*args, **kwargs):
                kwargs.pop("device", None)
                return fn(*args, **kwargs)
            return call
        e = engine.GlmEngine.__new__(engine.GlmEngine)
        e.world, e.rank = 2, 0
        e.torch = NS(tensor=cpu(torch.tensor), empty=cpu(torch.empty), zeros=cpu(torch.zeros), int32=torch.int32)
        calls = []
        def gather(send, receive):
            calls.append(send.numel())
            receive.copy_(torch.cat([send, send]))
        e.comm = NS(all_gather=gather)
        self.assertEqual(e._gather_ints([3, 8]), [[3, 8], [3, 8]])
        self.assertEqual(e._share([2, 4, 6]), [2, 4, 6])
        self.assertEqual(e._share([]), [])
        self.assertEqual(calls, [2, 1, 3, 1, 0])

    def test_calibration_singleton_has_no_collective(self):
        torch = self.torch
        e = engine.GlmEngine.__new__(engine.GlmEngine)
        e.world, e.rank, e.comm, e.load_mtp = 1, 0, None, True
        e.w, e.drafter = NS(cfg=NS(vocab=128), mtp=None), None
        observed = []
        st = NS(mtp_len=0, pos=64, mtp_drafted=0, set_mtp_len=lambda n: None)
        e.e = NS(st=st, pbuf=NS(fnormed=torch.zeros((8, 4))), forward=lambda ids: None, reset=lambda: observed.append("reset"))
        def tensor(*args, **kwargs):
            kwargs.pop("device", None)
            return torch.tensor(*args, **kwargs)
        e.torch = NS(tensor=tensor, float32=torch.float32, cuda=NS(synchronize=lambda: None))
        decode = ModuleType("tensorfold.families.glm5_next.cuda.decode")
        decode.__dict__.update(prefill=lambda *args, **kwargs: observed.append(kwargs["mtp"]),
                              draft=lambda *args, **kwargs: self.fail("calibrating absent MTP"))
        with patch.dict(sys.modules, {decode.__name__: decode}):
            costs = e._calibrate()
        self.assertEqual(observed, [False, "reset"])
        self.assertEqual(len(costs["verify"]), 8)
        self.assertEqual(costs["mtp"], 0)
        self.assertEqual(set(costs["timed"]), {"v" + str(n) for n in range(1, 9)})

    def test_full_reader_real_cpu_bytes_and_prefetch(self):
        torch = self.torch
        sys.path.insert(0, str(ROOT / "src"))
        with tempfile.TemporaryDirectory(dir=SCRATCH) as folder:
            names = ["model.language_model.layers.0.self_attn.qkv_proj.trellis",
                     "model.language_model.layers.0.self_attn.conv1d.weight",
                     "model.language_model.layers.3.mlp.experts.0.gate_proj.mul1", "lm_head.trellis"]
            raw = np.arange(30, dtype=np.uint8)
            split.write(str(Path(folder, "model.safetensors")), [(n, "I16", [3, 5], raw) for n in names], None)
            for prefetch in (False, True):
                rd = split.RankReader(folder, 0, world=1)
                try:
                    if prefetch: rd.prefetch(names)
                    for n in names:
                        value = rd.get(n)
                        self.assertEqual(tuple(value.shape), (3, 5))
                        self.assertEqual(value.dtype, torch.int16)
                        np.testing.assert_array_equal(value.view(torch.uint8).numpy().reshape(-1), raw)
                finally:
                    rd.close()


class GraphContracts(unittest.TestCase):
    def test_target_only_graphs_skip_mtp_and_keep_recurrence_parities(self):
        calls = []
        cuda = NS(graph_pool_handle=lambda: 1, synchronize=lambda: None, CUDAGraph=lambda: object(), graph=lambda *a, **k: nullcontext())
        env = {"torch": NS(cuda=cuda, no_grad=nullcontext), "prof": NS(active=True), "latent": NS(ENABLED=True),
               "compute": lambda w, st, b, r, **k: calls.append(tuple(st.cur)),
               "mtp_compute": lambda *a, **k: self.fail("MTP capture in target-only"),
               "pool_buckets": lambda n: [n]}
        Graphs = definitions(BASE / "cuda/graphs.py", {"Graphs"}, env)["Graphs"]
        e = NS(w=NS(mtp=object(), meta={"target_only": True}), st=NS(cur=[1, 0], index=[(None, None, NS(shape=(2050,)))]), buf=object(), mbuf=NS())
        graphs = Graphs(e, main_rows=(1,), mtp_rows=(1,))
        self.assertEqual(set(graphs.main), {(1, 0), (1, 1)})
        self.assertEqual(set(graphs.sparse), {(1, 0, 2048), (1, 1, 2048)})
        self.assertEqual(graphs.mtp, {})
        self.assertEqual(graphs.sparse_mtp, {})
        self.assertEqual(e.st.cur, [1, 0])


if __name__ == "__main__":
    unittest.main()
