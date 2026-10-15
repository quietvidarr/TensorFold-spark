"""The EXL3 prompt-window expert kernel (``routed(..., prompt=True)``) against the grouped decode kernel: the same
outputs to fp32 rounding, every member present, a row's result independent of the window it is in."""

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


def _picks(R, E, top, g, shared=True):
    routed = torch.rand((R, E - 1 if shared else E), device="cuda", generator=g).topk(top, dim=1).indices.int()
    if shared:
        routed = torch.cat([routed, torch.full((R, 1), E - 1, dtype=torch.int32, device="cuda")], 1)
    return routed.contiguous()


@pytest.mark.parametrize("R,k2s", [(64, [8]), (700, [8]), (2048, [8]), (300, [6, 8, 12, 16])])
def test_prompt_matches_grouped(R, k2s):
    E, D, I, top = 129, 2560, 640, 10
    ex, g = _layer(E, D, I, k2s)
    pick = _picks(R, E, top, g)
    slots = top + 1
    x = (0.5 * torch.randn((R, D), device="cuda", generator=g)).bfloat16()
    s = X.Scratch(ex, R, slots)
    ref = X.routed(x, pick, None, ex, s, None, R).clone()
    got = X.routed(x, pick, None, ex, s, None, R, prompt=True).clone()
    assert not torch.isnan(got).any()
    # the K sums agree to fp32 rounding; the gate/up epilogue then rounds to fp16 (half an fp16 ulp is 2.4e-4 of
    # the value), so an output moves by about that much where a sum lands near a rounding boundary
    err = (got - ref).abs().max() / ref.abs().max()
    assert err < 2e-3, f"prompt kernel off by {err:.2e}"
    rel = ((got - ref).norm() / ref.norm()).item()
    assert rel < 1e-4, f"prompt kernel off by {rel:.2e} (norm)"


def test_prompt_rows_independent_of_window():
    E, D, I, top = 129, 2560, 640, 10
    ex, g = _layer(E, D, I, [8])
    R = 512
    pick = _picks(R, E, top, g)
    x = (0.5 * torch.randn((R, D), device="cuda", generator=g)).bfloat16()
    s = X.Scratch(ex, R, top + 1)
    whole = X.routed(x, pick, None, ex, s, None, R, prompt=True).clone().view(R, top + 1, D)
    part = X.routed(x[100:300].contiguous(), pick[100:300].contiguous(), None, ex, s, None, 200,
                    prompt=True).clone().view(200, top + 1, D)
    assert torch.equal(whole[100:300], part)          # bit for bit: a row's sums are its own


def test_prompt_summed_is_the_weighted_slot_sum():
    E, D, I, top = 129, 2560, 640, 10
    ex, g = _layer(E, D, I, [8])
    R = 300
    pick = _picks(R, E, top, g)
    wts = torch.rand((R, top + 1), device="cuda", generator=g)
    x = (0.5 * torch.randn((R, D), device="cuda", generator=g)).bfloat16()
    s = X.Scratch(ex, R, top + 1)
    y = X.routed(x, pick, None, ex, s, None, R, prompt=True).clone().view(R, top + 1, D)
    out = torch.empty((R, D), dtype=torch.float32, device="cuda")
    X.routed(x, pick, wts, ex, s, out, R, prompt=True, summed=True)
    ref = torch.zeros((R, D), dtype=torch.float32, device="cuda")
    for k in range(top + 1):                          # slots in order, fp32, as the write-back sums them
        ref = torch.addcmul(ref, y[:, k], wts[:, k:k + 1])
    assert ((out - ref).abs().max() / ref.abs().max()).item() < 1e-6


def test_f16_prompt_path_matches_chunked():
    from tensorfold.families.qwen4_exp.cuda.exl3_mm import Scratch, f16

    sc = Scratch(11)
    for n, k in ((324, 10240), (10240, 320), (96, 2560)):
        m = f16(sc, [0.02 * torch.randn((n, k))], "cuda")
        sc.part = torch.empty((m.sk * 128 * n,), dtype=torch.float32, device="cuda")
        x = torch.randn((700, k), device="cuda").bfloat16()
        a = torch.empty((700, n), dtype=torch.float32, device="cuda")
        b = torch.empty_like(a)
        m(x, a)
        m.prefill(x, b)
        assert ((a - b).abs().max() / a.abs().max()).item() < 1e-4
        part = torch.empty((200, n), dtype=torch.float32, device="cuda")
        m.prefill(x[150:350].contiguous(), part)
        assert torch.equal(part, b[150:350])          # a row's sums do not depend on the piece


@pytest.mark.parametrize("k2s", [[8], [6, 8, 12, 16]])
def test_prompt_rotated_in_kernel_is_bit_identical(k2s):
    """bf16 inputs rotate inside the prompt kernel (prefill_rot); fp16 inputs go through rot_in: the same bits."""
    E, D, I, top = 129, 2560, 640, 10
    ex, g = _layer(E, D, I, k2s)
    R = 333
    pick = _picks(R, E, top, g)
    xb = (0.5 * torch.randn((R, D), device="cuda", generator=g)).bfloat16()
    s = X.Scratch(ex, R, top + 1)
    fused = X.routed(xb, pick, None, ex, s, None, R, prompt=True).clone()
    s.xg.zero_()
    ext = X._ext()
    ext.rot_in(xb, xb.stride(0), pick, ex.suh_g, ex.suh_u, s.xg, s.xu, R, D, top + 1, E)
    ids, members = s.window(R)
    P = R * (top + 1)
    ext.group_any(pick, ids, s.count, members, s.counts, s.slot_of, s.work, R, top + 1, E, X.PREFILL_PASS)
    z_ref = torch.zeros((2 * P * I,), dtype=torch.float32, device="cuda")
    z_rot = torch.zeros_like(z_ref)
    most = P // X.PREFILL_PASS + min(P, E)
    ext.prefill(s.xg, s.xu, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, ids, s.count, members, s.work, z_ref, 2, D,
                I, P, top + 1, ex.cb, ex.k2_gu[0], ex.k2_gu[1], most)
    ext.prefill_rot(xb, xb.stride(0), ex.suh_g, ex.suh_u, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, ids, s.count,
                    members, s.work, z_rot, D, I, P, top + 1, ex.cb, ex.k2_gu[0], ex.k2_gu[1], most)
    assert torch.equal(z_ref, z_rot)
    assert not torch.isnan(fused).any()


def test_upmix_f16_matches_up_then_mix():
    from tensorfold.families.qwen4_exp.cuda import glue
    from tensorfold.families.qwen4_exp.cuda.exl3_mm import Scratch, f16, upmix_f16

    S, D, low, R = 4, 2560, 320, 700
    sc = Scratch(11)
    up = f16(sc, [0.05 * torch.randn((S * D, low))], "cuda")
    act = torch.randn((R, low), device="cuda").bfloat16()
    normed = torch.randn((R, S * D), device="cuda").bfloat16()
    u = torch.empty((R, S * D), dtype=torch.bfloat16, device="cuda")
    up.prefill(act, u)
    mixed_ref = torch.empty((R, D), dtype=torch.bfloat16, device="cuda")
    xs_ref = torch.empty((R, D // 32), dtype=torch.float32, device="cuda")
    glue.hc_mix(u, normed, mixed_ref, xs_ref, S)
    mixed = torch.empty_like(mixed_ref)
    xs = torch.empty_like(xs_ref)
    assert upmix_f16(act, up, normed, mixed, xs, S)
    diff = (mixed.float() - mixed_ref.float()).abs()
    # the up sums may differ in the last fp32 bit from F16.prefill's tiles, which can flip a bf16 rounding
    assert diff.max().item() <= 2 * mixed_ref.float().abs().max().item() * 2 ** -8
    assert (diff > 0).float().mean().item() < 0.01
    assert torch.allclose(xs, xs_ref, rtol=1e-2, atol=1e-2)
