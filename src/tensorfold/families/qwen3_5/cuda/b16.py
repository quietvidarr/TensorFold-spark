"""A plain fp16/bf16 linear (``b16.cu``) for what an EXL3 pack leaves unquantized: one warp an output, row-invariant."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_qwen_b16_v4", sources=[str(here / "b16.cpp"), str(here / "b16.cu")],
                extra_cuda_cflags=["-O3"], verbose=False)


def matmul(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """x [M, K] @ w [N, K]^T (+ bias [N]), x cast to the weight's dtype when they differ."""

    if x.dtype != w.dtype:
        x = x.to(w.dtype)
    b = bias if bias is not None and bias.numel() else torch.empty(0, dtype=w.dtype, device=w.device)
    return _ext().b16_linear(x.contiguous(), w.contiguous(), b)


def prompt(x: torch.Tensor, w: torch.Tensor, bm: int = 0) -> torch.Tensor:
    """Prompt rows x [M, K] @ w [N, K]^T on the bf16 mma, one K chain a row: chunk-invariant; ``bm`` keeps bits."""

    if w.dtype != torch.bfloat16 or x.shape[1] % 64:
        return matmul(x, w)
    return _ext().b16_prompt(x.to(torch.bfloat16).contiguous(), w.contiguous(), bm)


def matmul_pair(x: torch.Tensor, w0: torch.Tensor, w1: torch.Tensor) -> list[torch.Tensor]:
    """x @ w0^T and x @ w1^T in one launch (the GDN gates b and a), each ``matmul``'s bits."""

    if w0.dtype != w1.dtype or x.shape[1] % 8:
        return [matmul(x, w0), matmul(x, w1)]
    return _ext().b16_linear_pair(x.to(w0.dtype).contiguous(), w0.contiguous(), w1.contiguous())


def prompt_pair(x: torch.Tensor, w0: torch.Tensor, w1: torch.Tensor) -> list[torch.Tensor]:
    """Prompt rows times two weights in one launch, each ``prompt``'s bits."""

    if w0.dtype != torch.bfloat16 or w1.dtype != torch.bfloat16 or x.shape[1] % 64:
        if w0.dtype == w1.dtype == torch.float16 and x.shape[0] >= 64 and x.shape[1] % 64 == 0 and x.stride(1) == 1:
            return [_prompt_f16(x, w0), _prompt_f16(x, w1)]
        return [prompt(x, w0), prompt(x, w1)]
    return _ext().b16_prompt_pair(x.to(torch.bfloat16).contiguous(), w0.contiguous(), w1.contiguous(), 0)


def _prompt_f16(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Prompt rows x [M, K] (bf16 or fp16) @ fp16 w [N, K]^T on the fp16 mma (x rounded to fp16 in the kernel, as
    matmul's cast does), fp16 out: fixed tiles and one K order a row, so a row never depends on its chunk (an EXL3
    pack's GDN gates: one pass over x instead of a cast copy and a warp-per-output launch)."""

    import triton

    from tensorfold.families.qwen4_exp.cuda.exl3_mm import _f16_mm

    m, k = x.shape
    n = w.shape[0]
    out = torch.empty((m, n), dtype=torch.float16, device=x.device)
    bn = 32 if n <= 32 else 64
    _f16_mm[(triton.cdiv(m, 32), triton.cdiv(n, bn), 1)](x, w.contiguous(), out, m, n, x.stride(0), n, K=k, KS=k,
                                                         BM=32, BN=bn, BK=64, F32=False, num_warps=4, num_stages=4)
    return out
