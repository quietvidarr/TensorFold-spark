"""Routed EXL3 experts of any codebook, a width per expert, one grouped launch a projection; rows never depend on the window."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import torch

CB_3INST, CB_MCG, CB_MUL1 = 0, 1, 2
ACT_BF16, ACT_F32 = 0, 1          # SwiGLU with the GLM family's bf16 roundings / in fp32
# Half-bits a value: 1..8 bits (2, 4, .. 16) and every half-integer rate 1.5..7.5 (3, 5, .. 15).
K2_SUPPORTED = tuple(range(2, 17))

# (n tiles a block, warps, K splits, tiles in flight): GLM's settings, whose arithmetic order this keeps bit for bit
GLM_GATEUP = (8, 4, 4, 1)
GLM_DOWN = (8, 4, 1, 1)


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    srcs = [str(here / f) for f in ("experts.cpp", "experts.cu", "experts_cb0.cu", "experts_cb1.cu", "experts_cb2.cu")]
    return load(name="tensorfold_exl3_experts_v4", sources=srcs, extra_cuda_cflags=["-O3", "-lineinfo"],
                verbose=False)


def codebook_id(name: str) -> int:
    """'3inst' / 'mcg' / 'mul1' (the checkpoint's quantization_config.codebook, or the marker tensor's name)."""

    ids = {"3inst": CB_3INST, "mcg": CB_MCG, "mul1": CB_MUL1}
    if name not in ids:
        raise ValueError(f"unknown EXL3 codebook {name!r}")
    return ids[name]


def k2_of(trellis: torch.Tensor) -> int:
    """Half-bits a value of a trellis int16 [K/16, N/16, 16 * K] (16 * K + 8 for the half-integer rates)."""

    w = trellis.shape[-1]
    if w % 8:
        raise ValueError(f"trellis last dim {w} is not a multiple of 8")
    k2 = w // 8
    if k2 not in K2_SUPPORTED:
        raise ValueError(f"unsupported EXL3 bit width {k2 / 2}")
    return k2


@dataclass
class Exl3RoutedExperts:
    """One layer's routed experts: trellis pointers and widths per projection, stacked suh/svh, the tensors keeping the trellises alive."""

    gate_ptr: torch.Tensor    # int64 [E]
    up_ptr: torch.Tensor
    down_ptr: torch.Tensor
    gate_k2: torch.Tensor     # int32 [E], half-bits a value
    up_k2: torch.Tensor
    down_k2: torch.Tensor
    suh_g: torch.Tensor       # fp16 [E, D]
    suh_u: torch.Tensor
    svh_g: torch.Tensor       # fp16 [E, I]
    svh_u: torch.Tensor
    suh_d: torch.Tensor       # fp16 [E, I]
    svh_d: torch.Tensor       # fp16 [E, D]
    count: int                # E
    dims: int                 # D (model width)
    width: int                # I (expert width on this rank)
    cb: int
    k2_gu: tuple[int, int]    # (min, max) K2 over gate and up
    k2_d: tuple[int, int]
    trellis_bytes: torch.Tensor   # int64 [E], gate + up + down trellis bytes of each expert (for GB/s)
    keep: list = field(default_factory=list, repr=False)

    def nbytes_read(self, ids: Sequence[int]) -> int:
        return int(self.trellis_bytes[list(ids)].sum())


def prepare(gate: Sequence[tuple], up: Sequence[tuple], down: Sequence[tuple], codebook: int | str,
            device="cuda") -> Exl3RoutedExperts:
    """A layer from per-expert (trellis, suh, svh) triples, trellises referenced in place, each at its own width."""

    cb = codebook_id(codebook) if isinstance(codebook, str) else int(codebook)
    E = len(gate)
    if not (len(up) == len(down) == E) or E == 0:
        raise ValueError("gate, up and down need the same, non-zero number of experts")
    D, I = gate[0][0].shape[0] * 16, gate[0][0].shape[1] * 16
    keep = []

    def table(mats, k, n):
        ptrs, k2s = [], []
        for t, _, _ in mats:
            if t.dtype != torch.int16 or not t.is_contiguous() or t.device.type != "cuda":
                raise ValueError("trellis must be a contiguous CUDA int16 tensor")
            if t.shape[0] * 16 != k or t.shape[1] * 16 != n:
                raise ValueError(f"expert shape {tuple(t.shape)} does not match [{k // 16}, {n // 16}, *]")
            k2s.append(k2_of(t))
            ptrs.append(t.data_ptr())
            keep.append(t)
        return (torch.tensor(ptrs, dtype=torch.int64, device=device),
                torch.tensor(k2s, dtype=torch.int32, device=device), k2s)

    gp, gk, gks = table(gate, D, I)
    upp, uk, uks = table(up, D, I)
    dp, dk, dks = table(down, I, D)

    def stack(mats, j, n):
        out = torch.empty((E, n), dtype=torch.float16, device=device)
        for e, m in enumerate(mats):
            out[e].copy_(m[j].reshape(-1))
        return out

    tb = torch.tensor([(D * I // 256) * (gks[e] + uks[e] + dks[e]) * 16 for e in range(E)], dtype=torch.int64)
    return Exl3RoutedExperts(gp, upp, dp, gk, uk, dk, stack(gate, 1, D), stack(up, 1, D), stack(gate, 2, I),
                             stack(up, 2, I), stack(down, 1, I), stack(down, 2, D), E, D, I, cb,
                             (min(gks + uks), max(gks + uks)), (min(dks), max(dks)), tb, keep)


def prepare_stacked(gt: torch.Tensor, ut: torch.Tensor, dt: torch.Tensor, suh_g, suh_u, svh_g, svh_u, suh_d, svh_d,
                    codebook: int | str) -> Exl3RoutedExperts:
    """A uniform-width layer stacked per projection: trellis [E, K/16, N/16, 16K] (int16, or GLM's int32 words), suh/svh [E, n]."""

    def as16(t):
        return t.view(torch.int16) if t.dtype == torch.int32 else t

    gt, ut, dt = as16(gt), as16(ut), as16(dt)
    E = gt.shape[0]
    return prepare([(gt[e], suh_g[e], svh_g[e]) for e in range(E)], [(ut[e], suh_u[e], svh_u[e]) for e in range(E)],
                   [(dt[e], suh_d[e], svh_d[e]) for e in range(E)], codebook, device=gt.device)


def default_config(K: int, N: int, gateup: bool) -> tuple[int, int, int, int]:
    """The tile setting for a K -> N projection (GLM's where it divides): the shape's alone, so rows stay independent."""

    cands = [GLM_GATEUP if gateup else GLM_DOWN, (8, 4, 2, 1), (8, 4, 1, 1), (4, 4, 2, 2), (4, 4, 1, 2)]
    for nt, w, sk, pf in cands:
        if K % (16 * sk * w) == 0 and N % (16 * nt) == 0:
            return nt, w, sk, pf
    raise ValueError(f"no tile setting divides K={K}, N={N}")


class Scratch:
    """Buffers for up to ``rows`` rows of ``slots`` slots; slots whose pick is not a routed expert are left to the caller."""

    def __init__(self, ex: Exl3RoutedExperts, rows: int, slots: int, cfg_gu=None, cfg_d=None, device="cuda") -> None:
        D, I = ex.dims, ex.width
        self.cfg_gu = cfg_gu or default_config(D, I, True)
        self.cfg_d = cfg_d or default_config(I, D, False)
        P = rows * slots
        self.xg = torch.zeros((P, D), dtype=torch.float16, device=device)
        self.xu = torch.zeros((P, D), dtype=torch.float16, device=device)
        self.xd = torch.zeros((P, I), dtype=torch.float16, device=device)
        # gate and up write 2 * splits * P * I partials, down splits * P * D
        self.z = torch.zeros((max(2 * self.cfg_gu[2] * I, self.cfg_d[2] * D) * P,), dtype=torch.float32, device=device)
        self.y = torch.zeros((P, D), dtype=torch.float32, device=device)
        maxu = min(P, ex.count)
        self.ids = torch.zeros((maxu,), dtype=torch.int32, device=device)
        # the prompt path's grouping (``group_any``): per-expert counts, list slots and its work list
        self.counts = torch.zeros((ex.count,), dtype=torch.int32, device=device)
        self.slot_of = torch.zeros((ex.count,), dtype=torch.int32, device=device)
        self.work = torch.zeros((ex.count + 1,), dtype=torch.int32, device=device)
        self.count = torch.zeros((1,), dtype=torch.int32, device=device)
        self.members_buf = torch.full((maxu * rows,), -1, dtype=torch.int32, device=device)
        self.rows, self.slots, self.count_experts = rows, slots, ex.count

    def window(self, R: int):
        """(ids, members) sized for R rows: the grids only span what R rows can use."""

        maxu = min(R * self.slots, self.count_experts)
        return self.ids[:maxu], self.members_buf[:maxu * R].view(maxu, R)


PREFILL_COLS, PREFILL_K, PREFILL_PASS = 320, 128, 128   # the prompt kernel's column strip, K split and members a program
PREFILL_COLS_NARROW = 256      # the strip a launch takes where 320 does not divide its N (Qwen3.6-35B-A3B: 512, 2048)


def prompt_kernel_fits(ex: Exl3RoutedExperts) -> bool:
    """The prompt kernel's shapes: gate/up and down split in its column strips and K steps."""

    D, I = ex.dims, ex.width
    strips = lambda n: n % PREFILL_COLS == 0 or n % PREFILL_COLS_NARROW == 0     # noqa: E731
    return D % PREFILL_K == 0 and I % PREFILL_K == 0 and strips(I) and strips(D)


def routed(x: torch.Tensor, pick: torch.Tensor, wts: torch.Tensor | None, ex: Exl3RoutedExperts, s: Scratch,
           out: torch.Tensor | None, R: int, limit: float = math.inf, act_mode: int = ACT_F32,
           group: bool = True, prompt: bool = False, summed: bool = False,
           out16: torch.Tensor | None = None) -> torch.Tensor:
    """Routed experts of R rows (picks >= E skipped): Y per slot, or ``out`` = the wts-weighted sum when ``wts``; no host sync.

    ``prompt``: a prompt window through the prompt kernel (each decoded weight tile against every member of its
    expert, the K sums in one order) when the shapes fit; same results to fp32 rounding, a row still its own.
    ``summed`` (with ``prompt``, ``wts`` and ``out``): only the weighted sum is written, not each slot's y."""

    ext = _ext()
    D, I, E = ex.dims, ex.width, ex.count
    slots = s.slots
    P = R * slots
    if R > s.rows:
        raise ValueError(f"{R} rows but the scratch holds {s.rows}")
    ids, members = s.window(R)
    if prompt and prompt_kernel_fits(ex):
        return _routed_prompt(x, pick, wts, ex, s, out, R, limit, act_mode, ids, members, summed)
    if group:
        ext.group(pick, ids, s.count, members, R, slots, E)
    ext.rot_in(x, x.stride(0), pick, ex.suh_g, ex.suh_u, s.xg, s.xu, R, D, slots, E)
    nt, w, sk, pf = s.cfg_gu
    ext.grouped(s.xg, s.xu, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, ids, s.count, members, s.z, 2, D, I,
                P, sk, slots, ex.cb, nt, w, pf, ex.k2_gu[0], ex.k2_gu[1])
    ext.gateup_epilogue(s.z, pick, ex.svh_g, ex.svh_u, ex.suh_d, s.xd, R, P, I, sk, slots, E, float(limit), act_mode)
    nt, w, sk, pf = s.cfg_d
    ext.grouped(s.xd, s.xd, ex.down_ptr, ex.down_ptr, ex.down_k2, ex.down_k2, ids, s.count, members, s.z, 1, I,
                D, P, sk, slots, ex.cb, nt, w, pf, ex.k2_d[0], ex.k2_d[1])
    if wts is None:
        ext.down_epilogue(s.z, pick, ex.svh_d, s.y, R, P, D, sk, slots, E)
        return s.y[:P]
    if out is None:
        out = torch.empty((R, D), dtype=torch.float32, device=x.device)
    # the down epilogue and the combine in one launch (the same arithmetic in the same order as the two)
    # ``out16``: the sum also as bf16 rows (torch's rounding), written by the same launch
    ext.down_combine(s.z, pick, ex.svh_d, s.y, wts, out, R, P, D, sk, slots, E, out16)
    return out


@lru_cache(maxsize=1)
def _ext2():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_exl3_prompt_moe_v4", sources=[str(here / "prompt_moe.cpp"), str(here / "prompt_moe.cu")],
                extra_cuda_cflags=["-O3", "-lineinfo"], verbose=False)


# The prompt path's kernels: "2" (default) the second generation (prompt_moe.cuh: gate/up with its epilogue in one
# launch, down with the slot sum in one), "1" the first (prefill_kernel and its epilogue kernels).
# TF_EXL3_PROMPT_DET, the second generation's summed output (all deterministic but 0):
#   1 (default) "grid": down adds each slot's term into out with fp32 red.add, every term first rounded to its row's
#     power-of-two grid (a bound on the row's sums from the xd norms and each down matrix's largest column norm), so
#     every addition is exact and the order cannot matter: bit-reproducible, a row's bits independent of the window,
#     at the speed of the plain red.add; rounding <= 2^-24 of the row's bound (see row_grid_kernel).
#   y: each slot's y [P, D] and a fixed-order slot sum (the first deterministic mode; ~1.3-1.5 ms a layer slower).
#   0: plain fp32 red.add (the order of the additions, so out's last bits, varies from run to run).
PROMPT_KERNEL = os.environ.get("TF_EXL3_PROMPT_KERNEL", "2")
PROMPT_DET = os.environ.get("TF_EXL3_PROMPT_DET", "1").strip().lower()
PROMPT_DET = {"0": "0", "fast": "0", "y": "y", "2": "y"}.get(PROMPT_DET, "grid")
P2_K2 = (4, 6, 8, 12, 16)              # trellis widths the second generation takes (2, 3, 4, 6 and 8 bits)
OUT_BUDGET = int(os.environ.get("TF_EXL3_OUT_BUDGET", str(8 << 20)))   # bytes of out the down launch adds into at once


def prompt2_fits(ex: Exl3RoutedExperts) -> bool:
    """The second generation's shapes: D and I in 128-blocks, gate and up of an expert at one width, every width 2, 3, 4, 6
    or 8 bits (2 and 3: GLM-5.3)."""

    ok = getattr(ex, "_p2_fits", None)
    if ok is None:
        widths = set(ex.gate_k2.tolist()) | set(ex.up_k2.tolist()) | set(ex.down_k2.tolist())
        ok = (ex.dims % 128 == 0 and ex.width % 128 == 0 and bool(torch.equal(ex.gate_k2, ex.up_k2))
              and widths <= set(P2_K2))
        ex._p2_fits = ok
    return ok


_XDT: dict = {}


def _xd_tiles(n: int, device) -> torch.Tensor:
    """gate_up's output, per work item [n] fp16 (down's input in k16 slices), a buffer a device, grown on demand."""

    buf = _XDT.get(device)
    if buf is None or buf.numel() < n:
        buf = _XDT[device] = torch.empty((n,), dtype=torch.float16, device=device)
    return buf


def down_colmax(ex: Exl3RoutedExperts, chunk: int = 32) -> torch.Tensor | None:
    """[E] fp32: each expert's largest column norm of its down map xd -> y as the prompt kernel applies it
    (W_q, then H128 a 128-column block, * svh / sqrt(128)); None when the trellises are not at hand. Computed once a
    layer (prompt_moe.cu colmax_kernel: fixed order, the same bits every run) and kept on ``ex``; only an upper
    bound is needed (row_grid_kernel adds 2^-6 for roundings)."""

    got = getattr(ex, "_p2_colmax", False)
    if got is not False:
        return got
    E, D, I = ex.count, ex.dims, ex.width
    downs = ex.keep[2 * E:3 * E] if len(ex.keep) >= 3 * E else None      # prepare keeps gate, up, down (then extras)
    if downs is None or any(t.shape[0] * 16 != I or t.shape[1] * 16 != D for t in downs):
        ex._p2_colmax = None
        return None
    dev = ex.svh_d.device
    sq = torch.empty((E, D // 128), dtype=torch.float32, device=dev)
    buf = torch.empty((chunk, I, D), dtype=torch.float16, device=dev)
    for e0 in range(0, E, chunk):
        n = min(chunk, E - e0)
        for j in range(n):
            t = downs[e0 + j]
            _ext().dequant(t.contiguous(), buf[j], k2_of(t), ex.cb)
        _ext2().colmax(buf, ex.svh_d[e0:e0 + n].contiguous(), sq[e0:e0 + n], n, I, D)
    cm = sq.amax(1).sqrt() * (1.0 / math.sqrt(128.0))
    ex._p2_colmax = cm
    return cm


_NRM: dict = {}


def _grid_bufs(P: int, G: int, R: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    nrm, grid = _NRM.get(device, (None, None))
    if nrm is None or nrm.numel() < P * G or grid.numel() < R:
        nrm = torch.empty((max(P * G, 1),), dtype=torch.float32, device=device)
        grid = torch.empty((max(R, 1),), dtype=torch.float32, device=device)
        _NRM[device] = (nrm, grid)
    return nrm, grid


def _routed_prompt2(x, pick, wts, ex: Exl3RoutedExperts, s: Scratch, out, R: int, limit: float, act_mode: int, ids,
                    members, summed: bool) -> torch.Tensor:
    ext, e2 = _ext(), _ext2()
    D, I, E = ex.dims, ex.width, ex.count
    slots = s.slots
    P = R * slots
    most = P // PREFILL_PASS + min(P, E)              # work items: passes of every expert's members
    ext.group_any(pick, ids, s.count, members, s.counts, s.slot_of, s.work, R, slots, E, PREFILL_PASS)
    items = torch.empty((most, 2), dtype=torch.int32, device=x.device)
    e2.item_table(s.work, s.count, items, most)
    cm = down_colmax(ex) if wts is not None and summed and PROMPT_DET == "grid" else None
    exact = cm is not None                             # red.add of terms on each row's grid: deterministic
    red = wts is not None and summed and (PROMPT_DET == "0" or exact)   # down adds each slot's w * y into out
    if red and out is None:
        out = torch.empty((R, D), dtype=torch.float32, device=x.device)
    e2.to_half(x, x.stride(0), s.xg, R, D, out if red else None)   # fp16(x * 2^-2) [R, D] in xg; out = 0
    xd = _xd_tiles(most * PREFILL_PASS * I, x.device)
    nrm, grid = _grid_bufs(P, I // 128, R, x.device) if exact else (None, None)
    e2.gate_up(s.xg, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.suh_g, ex.suh_u, ex.svh_g, ex.svh_u, ex.suh_d, ids,
               s.count, members, s.work, items, xd, D, I, P, slots, ex.cb, most, ex.k2_gu[1], float(limit), act_mode,
               nrm)
    if exact:
        e2.row_grid(nrm, pick, wts, cm, grid, R, slots, E, I // 128)
    if red:
        e2.down(xd, ex.down_ptr, ex.down_k2, ex.svh_d, ids, s.count, members, s.work, items, wts, out, D, I, slots,
                ex.cb, most, ex.k2_d[1], R, OUT_BUDGET, grid)
        return out
    # each slot's y; for the fixed-order sum in Z (free here), since a caller's out may be a view of s.y (Flash's is)
    y = s.z[:P * D].view(P, D) if wts is not None and summed else s.y
    e2.down(xd, ex.down_ptr, ex.down_k2, ex.svh_d, ids, s.count, members, s.work, items, None, y, D, I, slots, ex.cb,
            most, ex.k2_d[1], R, 0, None)
    if wts is None:
        return s.y[:P]
    if out is None:
        out = torch.empty((R, D), dtype=torch.float32, device=x.device)
    if summed:
        e2.slot_sum(y, pick, wts, out, R, D, slots, E)
    else:
        ext.combine(s.y, wts, out, R, D, slots)
    return out


def _routed_prompt(x, pick, wts, ex: Exl3RoutedExperts, s: Scratch, out, R: int, limit: float, act_mode: int, ids,
                   members, summed: bool = False) -> torch.Tensor:
    if (PROMPT_KERNEL != "1" and prompt2_fits(ex) and x.stride(1) == 1 and x.stride(0) % 8 == 0
            and x.data_ptr() % 16 == 0 and x.dtype in (torch.bfloat16, torch.float16)):
        return _routed_prompt2(x, pick, wts, ex, s, out, R, limit, act_mode, ids, members, summed)
    ext = _ext()
    D, I, E = ex.dims, ex.width, ex.count
    slots = s.slots
    P = R * slots
    most = P // PREFILL_PASS + min(P, E)              # work items: passes of every expert's members
    ext.group_any(pick, ids, s.count, members, s.counts, s.slot_of, s.work, R, slots, E, PREFILL_PASS)
    if x.dtype == torch.bfloat16 and x.stride(1) == 1 and x.stride(0) % 4 == 0:   # rotated in the kernel
        ext.prefill_rot(x, x.stride(0), ex.suh_g, ex.suh_u, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, ids,
                        s.count, members, s.work, s.z, D, I, P, slots, ex.cb, ex.k2_gu[0], ex.k2_gu[1], most)
    else:
        ext.rot_in(x, x.stride(0), pick, ex.suh_g, ex.suh_u, s.xg, s.xu, R, D, slots, E)
        ext.prefill(s.xg, s.xu, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, ids, s.count, members, s.work, s.z, 2,
                    D, I, P, slots, ex.cb, ex.k2_gu[0], ex.k2_gu[1], most)
    ext.gateup_epilogue(s.z, pick, ex.svh_g, ex.svh_u, ex.suh_d, s.xd, R, P, I, 1, slots, E, float(limit), act_mode)
    ext.prefill(s.xd, s.xd, ex.down_ptr, ex.down_ptr, ex.down_k2, ex.down_k2, ids, s.count, members, s.work, s.z, 1, I,
                D, P, slots, ex.cb, ex.k2_d[0], ex.k2_d[1], most)
    if wts is None:
        ext.down_epilogue(s.z, pick, ex.svh_d, s.y, R, P, D, 1, slots, E)
        return s.y[:P]
    if out is None:
        out = torch.empty((R, D), dtype=torch.float32, device=x.device)
    if summed:
        ext.down_sum(s.z, pick, ex.svh_d, wts, out, R, P, D, 1, slots, E)
    else:
        ext.down_combine(s.z, pick, ex.svh_d, s.y, wts, out, R, P, D, 1, slots, E)
    return out


def dequant(trellis: torch.Tensor, codebook: int | str) -> torch.Tensor:
    """W_q [K, N] fp16 of one matrix through the kernels' own lane decode (ExLlamaV3's ``reconstruct``); for tests."""

    cb = codebook_id(codebook) if isinstance(codebook, str) else int(codebook)
    k2 = k2_of(trellis)
    K, N = trellis.shape[0] * 16, trellis.shape[1] * 16
    out = torch.empty((K, N), dtype=torch.float16, device=trellis.device)
    _ext().dequant(trellis.contiguous(), out, k2, cb)
    return out
