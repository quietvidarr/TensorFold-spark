"""Packed TP1 carrier contracts. CPU allocation/source/dispatch, not GPU parity."""
import ast
import inspect
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
from tensorfold.families.glm5_next.cuda import forward, glue, packed, graphs, mtp

ROOT = Path(__file__).resolve().parents[3]
ORIGINALS = ROOT / 'residual-fp32/originals/TensorFold/src/tensorfold/families/glm5_next/cuda'


def functions(source):
    tree = ast.parse(source)
    return {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef)}


@pytest.mark.parametrize('world,meta,expected', [(1, {'packed_exl3': True}, True), (2, {'packed_exl3': True}, False),
    (1, {}, False), (2, {}, False), (1, {'packed_exl3': False}, False)])
def test_route_is_explicit_packed_tp1(world, meta, expected):
    assert packed.fp32_residual(NS(world=world, meta=meta)) is expected


class Spy:
    def __init__(self): self.calls = []
    def __getitem__(self, grid):
        def call(*args, **kw): self.calls.append((grid, args, kw))
        return call


@pytest.mark.parametrize('enabled', [False, True])
def test_dispatch_no_allocations_and_default_legacy(monkeypatch, enabled):
    old, new = Spy(), Spy()
    monkeypatch.setattr(glue, '_hc_post', old)
    monkeypatch.setattr(glue, '_hc_post_fp32', new)
    x = torch.empty(4, 512, dtype=torch.float32 if enabled else torch.bfloat16)
    g = torch.empty(1, 4, 128)
    p, c = torch.empty(4, 4), torch.empty(4, 16)
    pointers = [t.data_ptr() for t in (x, g, p, c)]
    def forbidden(*a, **kw): raise AssertionError('allocation during dispatch')
    monkeypatch.setattr(torch, 'empty', forbidden)
    for r in (1, 4, 1, 4):
        kwargs = {'fp32_residual': True} if enabled else {}
        glue.hc_post(x[:r], x[:r], g[:, :r], p[:r], c[:r], **kwargs)
    selected = new if enabled else old
    assert len(selected.calls) == 4 and not (old if enabled else new).calls
    assert pointers == [t.data_ptr() for t in (x, g, p, c)]
    assert inspect.signature(glue.hc_post).parameters['fp32_residual'].default is False
    if enabled:
        assert all(call[2]['enable_fp_fusion'] is False for call in new.calls)


@pytest.mark.parametrize('world,dtype', [(2, torch.float32), (1, torch.bfloat16)])
def test_fp32_dispatch_rejects_unsafe_contract(monkeypatch, world, dtype):
    new = Spy()
    monkeypatch.setattr(glue, '_hc_post_fp32', new)
    with pytest.raises(ValueError, match='FP32|TP1'):
        glue.hc_post(torch.empty(1,512,dtype=dtype), torch.empty(1,512,dtype=dtype),
                     torch.empty(world,1,128), torch.empty(1,4), torch.empty(1,16), fp32_residual=True)
    assert not new.calls


def test_kernel_source_native_order_and_no_casts():
    source = Path(glue.__file__).read_text()
    n = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == '_hc_post_fp32')
    text = ast.get_source_segment(source, n)
    assert 'v = ps * branch' in text
    ordered = ['v = ps * branch'] + [f'v = tl.fma(c{i}, x{i}, v)' for i in range(4)]
    assert [text.index(s) for s in ordered] == sorted(text.index(s) for s in ordered)
    assert text.count('tl.fma(') == 4
    assert 'tl.bfloat16' not in text and 'tl.float16' not in text
    assert 'tl.store(XOUT + r * (S * D) + s * D + d, v)' in text
    # All four input streams must be loaded before any output store (in-place safe).
    assert text.index('x3 = tl.load') < text.index('tl.store')


def test_legacy_kernels_and_mtp_graph_sources_unchanged():
    old = functions((ORIGINALS/'glue.py').read_text())
    new = functions(Path(glue.__file__).read_text())
    for name in old:
        if name not in ('hc_post', 'hc_pre'):
            assert old[name] == new[name], name
    old_text = (ORIGINALS/'glue.py').read_text()
    new_text = Path(glue.__file__).read_text()
    for name in ('_hc_post','_embed_b16','_hc_partial','_hc_finish','_stream_mean','_residual_add'):
        before = next(n for n in ast.parse(old_text).body if isinstance(n,ast.FunctionDef) and n.name==name)
        after = next(n for n in ast.parse(new_text).body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.get_source_segment(old_text,before)==ast.get_source_segment(new_text,after)
    for module in (mtp, graphs):
        assert (ORIGINALS/Path(module.__file__).name).read_bytes() == Path(module.__file__).read_bytes()


def make_weights(world=1, is_packed=True, has_mtp=False):
    c = NS(hidden=128, streams=4, heads=2, lin_heads=2, kv_lora=128, q_lora=128, qk_dim=128,
           v_dim=128, index_dim=128, index_heads=2, dense_width=256, top_k=1, moe_width=128,
           shared_width=128, experts=2, quant='exl3', mtp_layers=int(has_mtp))
    head = packed.PlainLinear(torch.ones(256//world,128,dtype=torch.float16))
    m = NS(layer=NS(moe=NS(experts=None))) if has_mtp else None
    return NS(cfg=c, device='cpu', world=world, meta={'packed_exl3': is_packed}, layers=[], head=head, mtp=m)


@pytest.mark.parametrize('world,is_packed,has_mtp,prefill', [(1,True,False,False), (1,True,True,False),
    (1,True,True,True), (1,False,False,False), (2,False,False,False)])
def test_buffers_keep_only_carrier_wide(monkeypatch, world, is_packed, has_mtp, prefill):
    # Actual Buffers allocations; only optional model projection enumeration is bounded to the fixture head.
    monkeypatch.setattr(packed, 'projections', lambda w: [w.head])
    monkeypatch.setattr(forward.latent, 'ENABLED', False)
    w = make_weights(world, is_packed, has_mtp)
    b = forward.Buffers(w, 4, capacity=16, prefill=prefill)
    enabled = is_packed and world == 1
    assert b.fp32_residual is enabled
    assert b.x.dtype == (torch.float32 if enabled else torch.bfloat16)
    assert b.x.numel()*b.x.element_size() == 4*4*128*(4 if enabled else 2)
    for name in ('normed','gu','act','eact','sgu','sact','hidden','fnormed','logits','lat','qr'):
        assert getattr(b, name).dtype == torch.bfloat16, name
    if has_mtp:
        for name in ('hin','me','mcat','mx'):
            assert getattr(b, name).dtype == torch.bfloat16, name
    else:
        assert b.hin is None and b.mx is None
    b.set_taps((0,1),128)
    assert all(t.dtype == torch.bfloat16 for t in b.taps)
    pointers = (b.x.data_ptr(), b.hcpart.data_ptr(), b.projection_scratch.xh.data_ptr()) if is_packed else (b.x.data_ptr(),)
    for r in (1,4,1,4): assert b.x[:r].data_ptr() == pointers[0]
    assert pointers[0] == b.x.data_ptr()


@pytest.mark.parametrize('load_mtp', [False, True])
def test_workspace_counts_full_carriers_not_microchunks(load_mtp):
    c = make_weights(has_mtp=True).cfg
    # Existing adapter projection/routed estimate plus extra bytes over legacy BF16 geometry.
    old_source = (ORIGINALS/'packed.py').read_text()
    namespace = {'__name__': packed.__name__, '__package__': packed.__package__}
    exec(compile(old_source, str(ORIGINALS/'packed.py'), 'exec'), namespace)
    old = namespace['workspace_bytes'](c, {}, rows=4, prefill_rows=513, load_mtp=load_mtp)
    delta = (4+513+(4 if load_mtp else 0))*4*128*2
    assert packed.workspace_bytes(c, {}, rows=4, prefill_rows=513, load_mtp=load_mtp) == old+delta


def test_layer_routes_both_updates(monkeypatch):
    seen=[]
    def post(*args, **kw): seen.append(kw['fp32_residual'])
    monkeypatch.setattr(glue, 'hc_post', post)
    monkeypatch.setattr(glue, 'hc_pre', lambda *a: None)
    monkeypatch.setattr(forward, 'kda_block', lambda *a: torch.empty(1,1,128))
    monkeypatch.setattr(forward, 'mlp_block', lambda *a: torch.empty(1,1,128))
    h=NS(fn=None,base=None,scale=None)
    layer=NS(attn_hc=h,ffn_hc=h,in_norm=None,post_norm=None,kind='kda',mlp=object())
    b=NS(x=torch.empty(1,512),normed=torch.empty(1,128),xs=torch.empty(1,2),post=torch.empty(1,4),
         comb=torch.empty(1,16),hcpart=torch.empty(1,16,32),fp32_residual=True)
    forward.layer_forward(layer, NS(cfg=NS(eps=1e-5,hc_eps=1e-6,hc_iters=20)), None,b,1)
    assert seen == [True,True]


def test_graph_capture_uses_preallocated_fp32_carrier(monkeypatch):
    from contextlib import nullcontext
    monkeypatch.setattr(forward.latent, 'ENABLED', False)
    w = make_weights()
    w.embed = torch.ones(256,128,dtype=torch.bfloat16)
    w.norm = torch.ones(128,dtype=torch.bfloat16)
    w.cfg.eps = 1e-5
    b = forward.Buffers(w,4,capacity=16)
    b.set_taps((),128)
    st = NS(cur=[],index=None,pos_dev=torch.zeros(1,dtype=torch.int32))
    seen=[]
    def embed(ids, table, d, copies, out):
        assert table.dtype == torch.bfloat16 and out.dtype == torch.float32
        assert out.data_ptr() == b.x.data_ptr()
        seen.append(out.shape[0])
    def mean(x,out):
        assert x.dtype == torch.float32 and out.dtype == torch.bfloat16
    monkeypatch.setattr(glue,'embed',embed)
    monkeypatch.setattr(glue,'stream_mean',mean)
    monkeypatch.setattr(glue,'rmsnorm',lambda *a: None)
    monkeypatch.setattr(forward,'mm',lambda *a: None)
    monkeypatch.setattr(torch.cuda,'graph_pool_handle',lambda: object())
    monkeypatch.setattr(torch.cuda,'synchronize',lambda: None)
    monkeypatch.setattr(torch.cuda,'CUDAGraph',lambda: object())
    monkeypatch.setattr(torch.cuda,'graph',lambda *a,**kw: nullcontext())
    before=(b.x.data_ptr(),b.hcpart.data_ptr(),b.projection_scratch.xh.data_ptr())
    def forbidden(*a,**kw): raise AssertionError('tensor allocation during capture')
    for name in ('empty','zeros','ones','empty_like','zeros_like','ones_like'):
        monkeypatch.setattr(torch,name,forbidden)
    g=graphs.Graphs(NS(w=w,st=st,buf=b),main_rows=(1,4),mtp_rows=())
    assert set(g.main)=={(1,0),(4,0)} and not g.mtp
    assert seen==[1,1,1,4,4,4]
    assert before==(b.x.data_ptr(),b.hcpart.data_ptr(),b.projection_scratch.xh.data_ptr())


def test_mtp_plain_residual_stays_bf16_not_mhc(monkeypatch):
    monkeypatch.setattr(packed,'projections',lambda w: [w.head])
    monkeypatch.setattr(forward.latent,'ENABLED',False)
    w=make_weights(has_mtp=True)
    w.embed=torch.ones(256,128,dtype=torch.bfloat16)
    w.cfg.eps=1e-5
    layer=NS(in_norm=None,post_norm=None,moe=NS(experts=None))
    w.mtp=NS(layer=layer,enorm=None,hnorm=None,eh=None,norm=None)
    w.draft_head=None
    b=forward.Buffers(w,4,capacity=16)
    g=torch.empty(1,1,128)
    residuals=[]
    def embed(ids,table,d,copies,out): assert out.dtype==torch.bfloat16 and copies==1
    def norm(x,weight,eps,out,*a): assert x.dtype==out.dtype==torch.bfloat16
    def mm(buf,x,q,xs,out): assert x.dtype==out.dtype==torch.bfloat16
    def residual(x,out,partials):
        assert x.dtype==out.dtype==torch.bfloat16 and partials is g
        residuals.append(x.data_ptr())
    monkeypatch.setattr(glue,'embed',embed)
    monkeypatch.setattr(glue,'rmsnorm',norm)
    monkeypatch.setattr(glue,'residual_add',residual)
    def forbidden(*a,**kw): raise AssertionError('MTP must not use mHC post')
    monkeypatch.setattr(glue,'hc_post',forbidden)
    monkeypatch.setattr(mtp,'mm',mm)
    monkeypatch.setattr(mtp.qmm,'group_sums',lambda x,out: out)
    monkeypatch.setattr(mtp,'dsa_block',lambda *a: g)
    monkeypatch.setattr(mtp,'moe_block',lambda *a: g)
    mtp.mtp_compute(w,NS(mtp_kc=None,mtp_vc=None,mtp_pos_dev=None,index=None),b,1)
    assert residuals==[b.mx.data_ptr(),b.mx.data_ptr()]
    assert b.x.dtype==torch.float32 and b.mx.dtype==torch.bfloat16


def test_embed_pre_mean_forward_fp32_pointer_without_conversion(monkeypatch):
    x=torch.empty(4,512,dtype=torch.float32)
    norm=torch.empty(4,128,dtype=torch.bfloat16)
    ids=torch.zeros(4,dtype=torch.int32)
    table=torch.ones(32,128,dtype=torch.bfloat16)
    embed,partial,finish,mean=Spy(),Spy(),Spy(),Spy()
    for name,spy in [('_embed_b16',embed),('_hc_partial',partial),('_hc_finish',finish),('_stream_mean',mean)]:
        monkeypatch.setattr(glue,name,spy)
    def forbidden(*a,**kw): raise AssertionError('allocating/casting carrier in wrapper')
    monkeypatch.setattr(torch,'empty',forbidden)
    glue.embed(ids,table,128,4,x)
    glue.hc_pre(x,None,None,None,table[0],norm,None,None,None,None,1e-5,1e-6,20)
    glue.stream_mean(x,norm)
    assert embed.calls[0][1][2] is x
    assert partial.calls[0][1][0] is x and finish.calls[0][1][0] is x
    assert mean.calls[0][1][0] is x and mean.calls[0][1][1] is norm
