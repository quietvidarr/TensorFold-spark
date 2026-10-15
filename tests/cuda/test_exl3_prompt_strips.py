"""The prompt expert kernel's 256-column strips (Qwen3.6-35B-A3B: expert width 512, model width 2048): the same
results as the grouped decode kernel, a row independent of its window; 320-column shapes keep their own strips."""

import pytest
import torch

from tensorfold.cuda.exl3 import experts as X

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _layer(E, D, I, k2s, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def mats(k, n):
        return [(torch.randint(-32768, 32767, (k // 16, n // 16, 8 * k2s[e % len(k2s)]), dtype=torch.int16,
                               device="cuda", generator=g),
                 (1 + 0.05 * torch.randn((k,), device="cuda", generator=g)).half(),
                 (1 + 0.05 * torch.randn((n,), device="cuda", generator=g)).half()) for e in range(E)]

    return X.prepare(mats(D, I), mats(D, I), mats(I, D), "mul1"), g


def _picks(R, E, top, g):
    routed = torch.rand((R, E - 1), device="cuda", generator=g).topk(top, dim=1).indices.int()
    return torch.cat([routed, torch.full((R, 1), E - 1, dtype=torch.int32, device="cuda")], 1).contiguous()


def test_strip_choice():
    ex, _ = _layer(9, 2048, 512, [12])
    assert X.prompt_kernel_fits(ex)
    ex, _ = _layer(9, 2048, 384, [12])                # 384: neither strip divides it
    assert not X.prompt_kernel_fits(ex)


# Qwen3.6-35B-A3B's 6.00bpw pack: routed experts 6-bit (k2 12), shared expert 8-bit (k2 16); 4-bit and mixed too
@pytest.mark.parametrize("R,k2s", [(64, [8]), (1000, [12]), (4096, [12, 12, 12, 16]), (300, [6, 8, 12, 16])])
def test_narrow_prompt_matches_grouped(R, k2s):
    E, D, I, top = 65, 2048, 512, 8
    ex, g = _layer(E, D, I, k2s)
    pick = _picks(R, E, top, g)
    x = (0.5 * torch.randn((R, D), device="cuda", generator=g)).bfloat16()
    s = X.Scratch(ex, R, top + 1)
    ref = X.routed(x, pick, None, ex, s, None, R).clone()
    got = X.routed(x, pick, None, ex, s, None, R, prompt=True).clone()
    assert not torch.isnan(got).any()
    assert ((got - ref).abs().max() / ref.abs().max()).item() < 2e-3
    # lab patch 0201: the second-generation prompt kernel is as close to the exact result as the grouped one
    from test_exl3_prompt_experts import reference
    exact = reference(ex, x, pick).view(R * (top + 1), D)
    e_prompt = ((got.double() - exact).norm() / exact.norm()).item()
    e_grouped = ((ref.double() - exact).norm() / exact.norm()).item()
    assert e_prompt < 1.05 * e_grouped + 1e-6, f"prompt kernel {e_prompt:.3e} vs grouped {e_grouped:.3e} from exact"


def test_narrow_rows_independent_of_window():
    E, D, I, top = 65, 2048, 512, 8
    ex, g = _layer(E, D, I, [12, 16])
    R = 512
    pick = _picks(R, E, top, g)
    wts = torch.rand((R, top + 1), device="cuda", generator=g)
    x = (0.5 * torch.randn((R, D), device="cuda", generator=g)).bfloat16()
    s = X.Scratch(ex, R, top + 1)
    det = X.PROMPT_DET
    X.PROMPT_DET = "grid"                             # deterministic (the default; "0" adds the slots freely)
    try:
        whole = torch.empty((R, D), dtype=torch.float32, device="cuda")
        X.routed(x, pick, wts, ex, s, whole, R, prompt=True, summed=True)
        part = torch.empty((200, D), dtype=torch.float32, device="cuda")
        X.routed(x[100:300].contiguous(), pick[100:300].contiguous(), wts[100:300].contiguous(), ex, s, part, 200,
                 prompt=True, summed=True)
    finally:
        X.PROMPT_DET = det
    assert torch.equal(whole[100:300], part)          # bit for bit
