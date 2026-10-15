"""Prompt-path speedups (lab patches 0203/0204): each one deterministic, independent of the window, and either bit-equal
to the kernels it replaces or as accurate."""

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")


def _suh(n, g):
    sign = torch.randint(0, 2, (n,), device="cuda", generator=g) * 2 - 1
    return (sign * (torch.rand(n, device="cuda", generator=g) + 0.5)).half()


def test_down_colmax_matches_fp64():
    """The exact sum's column bound: each expert's largest column norm of W_q -> H128 -> svh / sqrt(128)."""
    from tensorfold.cuda.exl3 import experts as X

    g = torch.Generator(device="cuda").manual_seed(3)
    E, D, I = 5, 512, 256

    def tr(k, n, bits):
        return torch.randint(-32768, 32767, (k // 16, n // 16, 16 * bits), dtype=torch.int16, device="cuda",
                             generator=g)

    def sc(n):
        return (torch.rand(n, device="cuda", generator=g) * 0.02 + 0.005).half()

    mats = lambda k, n, bits: [(tr(k, n, bits), sc(k), sc(n)) for _ in range(E)]   # noqa: E731
    ex = X.prepare(mats(D, I, 4), mats(D, I, 4), mats(I, D, 6), "mul1")
    cm = X.down_colmax(ex)
    i = torch.arange(128, device="cuda")
    h = (1.0 - 2.0 * torch.tensor([bin(v).count("1") & 1 for v in range(128)], device="cuda")[i[:, None] & i[None, :]]
         .double())
    for e in range(E):
        w = X.dequant(ex.keep[2 * E + e], "mul1").double().view(I, D // 128, 128) @ h
        ref = (w.square().sum(0).reshape(D) * ex.svh_d[e].double().square()).max().sqrt() / math.sqrt(128.0)
        assert abs(cm[e].item() - ref.item()) <= 1e-5 * ref.item()
    ex._p2_colmax = False
    assert torch.equal(X.down_colmax(ex), cm)                   # the same bits again


def test_chain_block_sizes_same_bits():
    from tensorfold.cuda.kernels import gdn

    g = torch.Generator(device="cuda").manual_seed(4)
    W, hk, hv = 300, 16, 32
    q = (torch.randn(W, hk, 128, device="cuda", generator=g) * 0.1).bfloat16()
    k = (torch.randn(W, hk, 128, device="cuda", generator=g) * 0.1).bfloat16()
    v = torch.randn(W, hv, 128, device="cuda", generator=g).bfloat16()
    gt = torch.rand(W, hv, device="cuda", generator=g) * 0.1 + 0.9
    beta = torch.rand(W, hv, device="cuda", generator=g)
    st = torch.randn(hv, 128, 128, device="cuda", generator=g) * 0.01
    import os

    got = []
    for rows in ("128", "64", "32"):
        os.environ["TF_GDN_CHAIN_ROWS"] = rows
        fin = torch.empty_like(st)
        got.append((gdn.chain(q, k, v, gt, beta, st, fin).clone(), fin))
    os.environ.pop("TF_GDN_CHAIN_ROWS")
    for y, fin in got[1:]:
        assert torch.equal(y, got[0][0]) and torch.equal(fin, got[0][1])


def test_gate_pair_fp16_prompt():
    """The GDN gates b, a (fp16 weights) on the fp16 mma: as accurate as matmul's, rows independent of the window."""
    from tensorfold.families.qwen3_5.cuda import b16

    g = torch.Generator(device="cuda").manual_seed(5)
    x = torch.randn(700, 2048, device="cuda", generator=g).bfloat16()
    w0 = (torch.randn(32, 2048, device="cuda", generator=g) * 0.02).half()
    w1 = (torch.randn(32, 2048, device="cuda", generator=g) * 0.02).half()
    got = b16.prompt_pair(x, w0, w1)
    part = b16.prompt_pair(x[100:400].contiguous(), w0, w1)
    for w, y, p in zip((w0, w1), got, part):
        assert y.dtype == torch.float16 and torch.equal(y[100:400], p)
        ref = x.double() @ w.double().t()
        old = b16.matmul(x, w).double()
        err, err_old = ((y.double() - ref).norm() / ref.norm()).item(), ((old - ref).norm() / ref.norm()).item()
        assert err <= 1.05 * err_old + 1e-6


def test_gate_pair_fp16_prompt_any_row_count():
    """A resumed prompt's tail can be a few rows: every row count (1..63 too) gives the rows of a wide call, so a
    resumed prompt equals one prefilled from its start (below 64 rows the pair once fell back to prompt())."""
    from tensorfold.families.qwen3_5.cuda import b16

    g = torch.Generator(device="cuda").manual_seed(6)
    x = torch.randn(300, 2048, device="cuda", generator=g).bfloat16()
    w0 = (torch.randn(32, 2048, device="cuda", generator=g) * 0.02).half()
    w1 = (torch.randn(32, 2048, device="cuda", generator=g) * 0.02).half()
    full = b16.prompt_pair(x, w0, w1)
    for a, n in ((0, 1), (7, 2), (33, 14), (100, 31), (200, 63), (237, 63)):
        part = b16.prompt_pair(x[a:a + n].contiguous(), w0, w1)
        for y, p in zip(full, part):
            assert torch.equal(y[a:a + n], p), (a, n)


def test_rotated_producers_match_rot_in():
    """gated_norm / gate_mul (qwen3_5), attn_gate and the DeltaNet back end (Flash Next) writing an EXL3 projection's
    rotated input: the bits of the bf16 output followed by rot_in."""
    from tensorfold.cuda.exl3.linear import _ext
    from tensorfold.families.qwen3_5.cuda import glue as g35
    from tensorfold.families.qwen4_exp.cuda import gdn_io
    from tensorfold.families.qwen4_exp.cuda import glue as g4

    g = torch.Generator(device="cuda").manual_seed(6)
    W = 300

    def rot(x, suh):
        out = torch.empty(x.shape, dtype=torch.float16, device="cuda")
        _ext().rot_in(x.contiguous(), suh, out)
        return out

    y = torch.randn(W, 8, 128, device="cuda", generator=g).bfloat16()
    z = torch.randn(W, 8, 128, device="cuda", generator=g).bfloat16()
    w = (torch.rand(128, device="cuda", generator=g) + 0.5).bfloat16()
    suh = _suh(8 * 128, g)
    xh = torch.empty(W, 8 * 128, dtype=torch.float16, device="cuda")
    assert g35.gated_norm_rot(y, z, w, 1e-6, suh, xh)
    assert torch.equal(xh.view(torch.int16), rot(g35.gated_norm(y, z, w, 1e-6)[0], suh).view(torch.int16))

    o = torch.randn(W, 4, 256, device="cuda", generator=g).bfloat16()
    qg = torch.randn(W, 4 * 512, device="cuda", generator=g).bfloat16()
    suh = _suh(4 * 256, g)
    xh = torch.empty(W, 4 * 256, dtype=torch.float16, device="cuda")
    assert g35.gate_mul_rot(o, qg, suh, xh, heads=4, head_dim=256)
    ref = rot(g35.gate_mul(o, qg, heads=4, head_dim=256)[0], suh)
    assert torch.equal(xh.view(torch.int16), ref.view(torch.int16))

    pa = torch.randn(W, 4 * 512 + 64, device="cuda", generator=g).bfloat16()
    gated = torch.empty(W, 4 * 256, dtype=torch.bfloat16, device="cuda")
    xs = torch.empty(W, 4 * 256 // 32, device="cuda")
    g4.attn_gate(o, pa, gated, xs, q_heads=4, head_dim=256)
    assert g4.attn_gate_rot(o, pa, suh, xh, q_heads=4, head_dim=256)
    assert torch.equal(xh.view(torch.int16), rot(gated, suh).view(torch.int16))

    NK, NV = 16, 48
    C = 2 * NK * 128 + NV * 128
    yb = torch.randn(W, NV, 128, device="cuda", generator=g).bfloat16()
    P = torch.randn(W, C + NV * 128 + 2 * NV, device="cuda", generator=g).bfloat16()
    out = torch.empty(W, NV * 128, dtype=torch.bfloat16, device="cuda")
    xs = torch.empty(W, NV * 4, device="cuda")
    gdn_io.back(yb, P, w, 1e-6, out, xs)
    suh = _suh(NV * 128, g)
    xh = torch.empty(W, NV * 128, dtype=torch.float16, device="cuda")
    gdn_io.back(yb, P, w, 1e-6, out, xs, suh, xh)
    assert torch.equal(xh.view(torch.int16), rot(out, suh).view(torch.int16))


def test_prefill_matmul_rotated_input():
    from tensorfold.cuda.exl3 import prefill as pf
    from tensorfold.cuda.exl3.linear import Exl3Linear, _ext

    g = torch.Generator(device="cuda").manual_seed(7)
    K, N = 512, 384
    tr = torch.randint(-32768, 32767, (K // 16, N // 16, 96), dtype=torch.int16)
    lin = Exl3Linear.from_tensors(tr, _suh(K, g).cpu(), (torch.rand(N) * 0.02 + 0.005).half(), "mul1")
    x = torch.randn(300, K, device="cuda", generator=g).bfloat16()
    ws = pf.Workspace()
    a = pf.matmul(lin, x, torch.empty(300, N, dtype=torch.bfloat16, device="cuda"), ws).clone()
    xh = pf.rotated_input(ws, 300, K, x.device)
    _ext().rot_in(x, lin.suh, xh)
    b = pf.matmul(lin, None, torch.empty(300, N, dtype=torch.bfloat16, device="cuda"), ws, xh=xh)
    assert torch.equal(a, b)


@pytest.mark.parametrize("mode", [1, 3])
def test_hc_readout_on_the_fly_same_bytes(mode):
    """Flash Next's prompt HC read-out without normed rows: write-back, down, act, up-mix byte for byte."""
    from tensorfold.families.qwen4_exp.cuda import glue
    from tensorfold.families.qwen4_exp.cuda.exl3_mm import (F16, f16_prefill_normed, f16_split, upmix_f16,
                                                             upmix_f16_normed)
    from tensorfold.families.qwen4_exp.cuda.hc_fused import write_norm

    g = torch.Generator(device="cuda").manual_seed(8 + mode)
    S, D, LOW, R = 4, 2560, 320, 300
    Wd = S * D
    down = F16((torch.randn(LOW + S, Wd, device="cuda", generator=g) * 0.01).half(), LOW + S, Wd,
               f16_split(LOW + S, Wd), None)
    up = F16((torch.randn(Wd, LOW, device="cuda", generator=g) * 0.05).half(), Wd, LOW, f16_split(Wd, LOW), None)
    scale = torch.rand(Wd, device="cuda", generator=g) + 0.5
    h0 = torch.randn(R, Wd, device="cuda", generator=g).bfloat16()
    inj = torch.rand(R, S, device="cuda", generator=g).bfloat16()
    branch = (torch.randn(R, D, device="cuda", generator=g).bfloat16() if mode == 1
              else torch.randn(1, R, D, device="cuda", generator=g))

    def bufs():
        bf = torch.bfloat16
        return dict(h=h0.clone(), pss=torch.empty(R, D // 256, S, device="cuda"),
                    normed=torch.empty(R, Wd, device="cuda", dtype=bf), xsn=torch.empty(R, Wd // 32, device="cuda"),
                    dn=torch.empty(R, LOW + S, device="cuda", dtype=bf), act=torch.empty(R, LOW, device="cuda", dtype=bf),
                    xsa=torch.empty(R, LOW // 32, device="cuda"), injo=torch.empty(R, S, device="cuda", dtype=bf),
                    mixed=torch.empty(R, D, device="cuda", dtype=bf), xsm=torch.empty(R, D // 32, device="cuda"),
                    rinv=torch.empty(R, S, device="cuda"))

    A, B = bufs(), bufs()
    write_norm(A["h"], A["pss"], scale, A["normed"], A["xsn"], S, 1e-6, mode, branch=branch, inject=inj)
    down.prefill(A["normed"], A["dn"])
    glue.hc_act(A["dn"], A["act"], A["xsa"], A["injo"], S, LOW)
    upmix_f16(A["act"], up, A["normed"], A["mixed"], A["xsm"], S)
    write_norm(B["h"], B["pss"], scale, None, None, S, 1e-6, mode, branch=branch, inject=inj, rinv=B["rinv"])
    assert f16_prefill_normed(down, B["h"], B["rinv"], scale, S, B["dn"])
    glue.hc_act(B["dn"], B["act"], B["xsa"], B["injo"], S, LOW)
    assert upmix_f16_normed(B["act"], up, B["h"], B["rinv"], scale, B["mixed"], B["xsm"], S)
    for key in ("h", "pss", "dn", "act", "xsa", "injo", "mixed", "xsm"):
        assert torch.equal(A[key].view(torch.uint8), B[key].view(torch.uint8)), key
