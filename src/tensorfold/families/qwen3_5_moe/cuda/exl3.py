"""Qwen3.6 MoE from an EXL3 pack: the 27B's EXL3 layers, routed experts (the shared one last) on the grouped EXL3 kernels.

The dense layers load as the 27B's (``qwen3_5/cuda/exl3_load.py``); each MLP's 256 experts and its shared expert go into
one ``tensorfold.cuda.exl3.experts`` table (each expert at its own width), read as Flash Next's are
(``qwen4_exp/cuda/exl3.py`` ``expert_table``). Routing is the MLX path's (``tensorfold.cuda.moe``: fp32 router logits,
top 8 renormalized, the shared expert as expert 256 with its sigmoid gate), so a row's bits never depend on the window:
verify rows use the grouped decode kernel, prompt chunks the prompt kernel (bits chunk-invariant, as for every EXL3
prompt matmul). The pack's own MTP layer (EXL3, ``mtp.*``) drafts; its draft head is the target head's rows for the
draft vocabulary, decoded once and requantized to MLX 4-bit (drafts only pick rows to verify, never output bits)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch

GS = 64              # the draft head's MLX groups
SCRATCH_ROWS = max(4096, int(__import__("os").environ.get("TF_PREFILL_CHUNK", "4096")))  # most rows one expert call takes: a prompt chunk (``qwen3_5/cuda/prefill.py`` CHUNK)
HEAD_ROWS = 128      # rows above which the MTP layer's experts take the prompt kernel (verify windows: at most 128)


class Buffers:
    """Routing scratch and the grouped kernels' buffers, shared by every layer (calls run in order on one stream)."""

    def __init__(self) -> None:
        self.rows = 0

    def allocate(self, experts, top_k: int, device) -> None:
        from tensorfold.cuda.exl3 import experts as x3

        if self.rows:
            return
        self.rows, self.slots = SCRATCH_ROWS, top_k + 1
        self.logits = torch.empty((self.rows, experts.count), dtype=torch.float32, device=device)  # E routed + shared gate
        self.pick = torch.empty((self.rows, self.slots), dtype=torch.int32, device=device)
        self.wts = torch.empty((self.rows, self.slots), dtype=torch.float32, device=device)
        self.out = torch.empty((self.rows, experts.dims), dtype=torch.float32, device=device)
        self.x3 = x3.Scratch(experts, self.rows, self.slots, device=device)


def scratch_bytes(text: dict) -> int:
    """``Buffers``' bytes for a config (the admission's fixed scratch): routing, x3.Scratch's inputs, partials and y."""

    from tensorfold.cuda.exl3.experts import default_config

    d, i = int(text["hidden_size"]), int(text["moe_intermediate_size"])
    experts, slots = int(text["num_experts"]) + 1, int(text["num_experts_per_tok"]) + 1
    rows = SCRATCH_ROWS
    pairs = rows * slots
    sk_gu, sk_d = default_config(d, i, True)[2], default_config(i, d, False)[2]
    x3 = pairs * (2 * d * 2 + i * 2 + d * 4) + max(2 * sk_gu * i, sk_d * d) * pairs * 4 \
        + min(pairs, experts) * (rows + 1) * 4 + 4 * experts * 4
    return x3 + rows * (experts * 4 + slots * 8 + d * 4)


@dataclass
class Exl3Routed:
    """A layer's router rows [E + 1, D] bf16 (the shared expert's gate row last) and its E + 1 EXL3 experts."""

    router: torch.Tensor
    experts: object                  # tensorfold.cuda.exl3.experts.Exl3RoutedExperts, the shared expert last
    top_k: int
    buffers: Buffers
    wide_prompt: bool = False        # the MTP layer: rows past HEAD_ROWS (a prompt's) take the prompt kernel too

    @property
    def count(self) -> int:
        return self.router.shape[0] - 1

    def run(self, x: torch.Tensor, *, prefill: bool = False) -> torch.Tensor:
        """x [R, D] bf16 -> [R, D] bf16 (``tensorfold.cuda.moe.run``'s contract): the top-k experts' renormalized,
        weighted sum plus the shared expert's, the slots summed in slot order in fp32 and rounded once."""

        from tensorfold.cuda import moe
        from tensorfold.cuda.exl3.experts import prompt_kernel_fits, routed

        b = self.buffers
        x = x.contiguous()
        rows = x.shape[0]
        prompt = (prefill or (self.wide_prompt and rows > HEAD_ROWS)) and prompt_kernel_fits(self.experts)
        out = torch.empty((rows, x.shape[1]), dtype=torch.bfloat16, device=x.device)
        for r0 in range(0, rows, b.rows):                # rows are independent: windows keep each row's bits
            n = min(b.rows, rows - r0)
            xr = x[r0:r0 + n]
            moe.router(xr, self.router, b.logits[:n])
            moe.select_rows(b.logits[:n], b, self.top_k, self.count)
            if prompt:
                routed(xr, b.pick[:n], b.wts[:n], self.experts, b.x3, b.out[:n], n, prompt=True, summed=True)
            else:
                routed(xr, b.pick[:n], b.wts[:n], self.experts, b.x3, b.out[:n], n)
            out[r0:r0 + n].copy_(b.out[:n])
        return out


def _routed(pk, prefix: str, top_k: int, count: int, buffers: Buffers, device, *, wide_prompt: bool = False):
    """``prefix`` (``...mlp``): its router and shared-expert gate rows (stored bf16) and its experts' table."""

    from tensorfold.families.qwen4_exp.cuda.exl3 import expert_table

    router = torch.cat([pk.get(prefix + ".gate.weight").to(torch.bfloat16).reshape(count, -1),
                        pk.get(prefix + ".shared_expert_gate.weight").to(torch.bfloat16).reshape(1, -1)])
    experts = expert_table(pk, prefix + ".experts", count, prefix + ".shared_expert", device)
    pk.release()
    buffers.allocate(experts, top_k, device)
    return Exl3Routed(router.to(device).contiguous(), experts, int(top_k), buffers, wide_prompt)


def load(model_dir: str | Path, device: str = "cuda"):
    """The pack's language model: dense layers as the 27B's EXL3 pack, each MLP an ``Exl3Routed``."""

    from tensorfold.families.qwen3_5.cuda.exl3_load import load_exl3
    from tensorfold.families.qwen3_5.cuda.weights import Config
    from tensorfold.families.qwen4_exp.cuda.exl3_pack import Pack

    model_dir = Path(model_dir)
    cfg = Config.read(model_dir)
    if not cfg.experts or not cfg.top_k:
        raise ValueError(f"{model_dir} is not a MoE checkpoint (num_experts / num_experts_per_tok)")
    pk = Pack(model_dir)
    buffers = Buffers()
    w = load_exl3(model_dir, device,
                  moe=lambda prefix: _routed(pk, prefix, cfg.top_k, cfg.experts, buffers, device))
    w.exl3_buffers = buffers
    torch.cuda.empty_cache()
    return w


def load_mtp(model_dir: str | Path, w, device: str = "cuda"):
    """The pack's MTP layer (``mtp.*``, EXL3) packed like the model's own layers, or None if the pack has none."""

    from tensorfold.cuda.exl3.linear import Exl3Linear
    from tensorfold.families.qwen3_5.cuda.weights import Attention, Exl3
    from tensorfold.families.qwen4_exp.cuda.exl3_pack import Pack

    from .weights import MTP

    pk = Pack(model_dir)
    if not pk.has("mtp.fc.trellis"):
        return None

    def x3(name: str) -> Exl3:
        layer = Exl3Linear.from_tensors(pk.get(name + ".trellis"), pk.scales(name, "su", "suh"),
                                        pk.scales(name, "sv", "svh"), pk.codebook(name),
                                        pk.get(name + ".bias") if pk.has(name + ".bias") else None, device=device)
        return Exl3(layer)

    def norm(name: str) -> torch.Tensor:
        """A centred RMSNorm weight (stored as gamma - 1, as the pack's other norms) with its 1 back."""

        t = pk.get(name)
        return (t.float() + 1.0).to(t.dtype).to(device).contiguous()

    p = "mtp.layers.0."
    attn = Attention(q=x3(p + "self_attn.q_proj"), k=x3(p + "self_attn.k_proj"), v=x3(p + "self_attn.v_proj"),
                     o=x3(p + "self_attn.o_proj"), q_norm=norm(p + "self_attn.q_norm.weight"),
                     k_norm=norm(p + "self_attn.k_norm.weight"))
    c = w.config
    moe = _routed(pk, p + "mlp", c.top_k, c.experts, w.exl3_buffers, device, wide_prompt=True)
    m = MTP(norm_e=norm("mtp.pre_fc_norm_embedding.weight"), norm_h=norm("mtp.pre_fc_norm_hidden.weight"),
            fc_e=None, fc_h=None, input_norm=norm(p + "input_layernorm.weight"),
            post_norm=norm(p + "post_attention_layernorm.weight"), attn=attn, moe=moe, norm=norm("mtp.norm.weight"),
            fc=x3("mtp.fc"))
    pk.release()
    return m


@torch.no_grad()
def draft_head(head, ids: torch.Tensor):
    """The EXL3 head's rows ``ids`` decoded through its own linear (fp32), as an MLX 4-bit g64 ``QLinear`` (tiled)."""

    from tensorfold.families.qwen3_5.cuda.qmm_fast import tile
    from tensorfold.families.qwen3_5.cuda.weights import QLinear

    k, device, step = head.k, ids.device, 128
    rows = torch.empty((len(ids), k), dtype=torch.float32, device=device)
    eye = torch.eye(step, dtype=torch.float32, device=device)
    for k0 in range(0, k, step):
        x = torch.zeros((step, k), dtype=torch.float32, device=device)
        x[:, k0:k0 + step] = eye
        rows[:, k0:k0 + step] = head(x)[:, ids].t()
    g = rows.view(len(ids), k // GS, GS)
    lo, hi = g.amin(dim=-1), g.amax(dim=-1)
    scale = ((hi - lo) / 15).clamp_min(1e-8).to(torch.bfloat16)
    bias = lo.to(torch.bfloat16)
    q = torch.round((g - bias.float()[..., None]) / scale.float()[..., None]).clamp(0, 15).to(torch.int64)
    words = (q.view(len(ids), k // 8, 8) << (torch.arange(8, device=device, dtype=torch.int64) * 4)).sum(dim=-1)
    words = words & 0xFFFFFFFF
    words = torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)
    del rows, g, q
    return tile(QLinear(words.contiguous(), scale.contiguous(), bias.contiguous(), gs=GS, bits=4))


def admission(geometry, depth: int):
    """An EXL3 pack's admission: the geometry plus the experts' and prompt matmuls' scratch, and its tensors' bytes
    as stored (vision tower skipped; the MTP layer counted when it drafts)."""

    from tensorfold.cuda.capacity import SIZES
    from tensorfold.cuda.geometry import exl3_weights, exl3_workspace, with_fixed

    def fixed(text: dict) -> int:
        d = int(text["hidden_size"])
        conv = 2 * int(text["linear_num_key_heads"]) * int(text["linear_key_head_dim"]) + \
            int(text["linear_num_value_heads"]) * int(text["linear_value_head_dim"])
        widest = max(conv, 2 * int(text["num_attention_heads"]) * int(text.get("head_dim") or 256))
        return scratch_bytes(text) + exl3_workspace(d * widest, SCRATCH_ROWS, max(d, widest))

    def weights(name: str, info: dict) -> tuple[int, int]:
        if name.startswith("mtp."):
            return (math.prod(info["shape"]) * SIZES[info["dtype"]] if depth else 0), 0
        return exl3_weights(name, info)

    return (lambda text: with_fixed(geometry(text), fixed(text))), weights
