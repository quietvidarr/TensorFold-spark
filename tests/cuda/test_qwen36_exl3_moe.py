"""Qwen3.6 MoE's EXL3 experts (``families/qwen3_5_moe/cuda/exl3.py`` ``Exl3Routed``) through ``tensorfold.cuda.moe.run``:
a verify row's bits never depend on the window (decode kernel), a prompt chunk's rows neither (prompt kernel), and the
two kernels agree to rounding."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _routed(E=64, D=2048, I=512, top=8, seed=0, wide_prompt=False):
    from tensorfold.cuda.exl3 import experts as X
    from tensorfold.families.qwen3_5_moe.cuda.exl3 import Buffers, Exl3Routed

    g = torch.Generator(device="cuda").manual_seed(seed)

    def mats(k, n, count, k2):
        return [(torch.randint(-32768, 32767, (k // 16, n // 16, 8 * k2), dtype=torch.int16, device="cuda",
                               generator=g),
                 (0.05 * torch.randn((k,), device="cuda", generator=g)).half(),
                 (0.05 * torch.randn((n,), device="cuda", generator=g)).half()) for _ in range(count)]

    gate, up, down = mats(D, I, E, 12) + mats(D, I, 1, 16), mats(D, I, E, 12) + mats(D, I, 1, 16), \
        mats(I, D, E, 12) + mats(I, D, 1, 16)                      # 6-bit routed experts, an 8-bit shared one
    ex = X.prepare(gate, up, down, "mul1")
    b = Buffers()
    b.allocate(ex, top, "cuda")
    router = (0.05 * torch.randn((E + 1, D), device="cuda", generator=g)).bfloat16()
    return Exl3Routed(router, ex, top, b, wide_prompt), g


def test_verify_rows_independent_of_window():
    from tensorfold.cuda import moe

    m, g = _routed()
    x = torch.randn((128, 2048), device="cuda", generator=g).bfloat16()
    whole = moe.run(x, m)
    for a, b in ((0, 1), (5, 21), (64, 128)):
        assert torch.equal(moe.run(x[a:b].contiguous(), m), whole[a:b])


def test_prompt_rows_independent_of_window_and_close_to_decode():
    from tensorfold.cuda import moe

    from tensorfold.cuda.exl3 import experts as X

    m, g = _routed()
    x = torch.randn((3000, 2048), device="cuda", generator=g).bfloat16()
    det = X.PROMPT_DET
    X.PROMPT_DET = "grid"                             # deterministic (the default; "0" adds the slots freely)
    try:
        prompt = moe.run(x, m, prefill=True)
        assert torch.equal(moe.run(x[1000:2500].contiguous(), m, prefill=True), prompt[1000:2500])
    finally:
        X.PROMPT_DET = det
    X.PROMPT_DET = "0"
    try:
        free = moe.run(x, m, prefill=True)            # bf16 roundings of fp32 sums in another order: rare 1-ulp flips
    finally:
        X.PROMPT_DET = det
    assert ((free.float() - prompt.float()).abs() > 0).float().mean().item() < 0.02
    decode = moe.run(x[:128].contiguous(), m)
    rel = ((prompt[:128].float() - decode.float()).norm() / decode.float().norm()).item()
    assert rel < 5e-3, f"prompt and decode kernels differ by {rel:.2e}"


def test_head_takes_prompt_kernel_for_wide_rows():
    from tensorfold.cuda import moe

    from tensorfold.cuda.exl3 import experts as X

    m, g = _routed(wide_prompt=True)
    x = torch.randn((600, 2048), device="cuda", generator=g).bfloat16()
    det = X.PROMPT_DET
    X.PROMPT_DET = "grid"                             # deterministic (the default; "0" adds the slots freely)
    try:
        assert torch.equal(moe.run(x, m), moe.run(x, m, prefill=True))
    finally:
        X.PROMPT_DET = det
