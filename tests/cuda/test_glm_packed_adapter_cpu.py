"""CPU contracts; no extension load and no claim of CUDA packed arithmetic."""
import inspect
from types import SimpleNamespace
import pytest
import torch
from tensorfold.cuda.exl3.linear import Exl3Linear
from tensorfold.families.glm5_next.cuda.packed import (
    PlainLinear, ComposedLinear, ProjectionScratch, matmul, resident_transform,
)
from tensorfold.families.glm5_next.cuda.weights import load


def test_load_contract():
    sig = inspect.signature(load)
    assert sig.parameters['world'].default == 2
    assert sig.parameters['load_mtp'].default is True


@pytest.mark.parametrize('bits', [2, 3, 4, 5])
def test_stored_trellis_preserved(bits):
    t = torch.arange(8 * 16 * 16 * bits, dtype=torch.int16).reshape(8,16,16*bits)
    u, v = torch.ones(128, dtype=torch.float16), torch.ones(256, dtype=torch.float16)
    q = Exl3Linear.from_tensors(t,u,v,'mul1',device='cpu',layout='stored')
    assert q.words.data_ptr() == t.data_ptr()
    assert q.bits == bits and (q.k,q.n)==(128,256)
    assert torch.equal(q.words.view(torch.int16),t)


def test_plain_dtype_and_composition_real_numeric():
    # FP16 value that a BF16 weight conversion would change.
    w = torch.tensor([[1.00390625, 0], [0, 2.00390625]],dtype=torch.float16)
    a=PlainLinear(w)
    b=PlainLinear(torch.tensor([[3.125, -0.25]],dtype=torch.float32))
    q=ComposedLinear([a,b])
    x=torch.tensor([[2.,3.],[-1.,4.]],dtype=torch.bfloat16)
    out=torch.empty(2,3,dtype=torch.float32)
    s=ProjectionScratch([q],2,'cpu')
    assert a.weight.dtype==torch.float16 and a.weight.data_ptr()==w.data_ptr()
    assert matmul(q,x,out,s).data_ptr()==out.data_ptr()
    expected=torch.cat([x.float()@w.float().T,x.float()@b.weight.T],dim=-1)
    torch.testing.assert_close(out,expected,rtol=0,atol=0)
    assert q.offsets==(0,2,3)
    assert not torch.equal(out[:,:2],x.float()@w.bfloat16().float().T)


def test_projection_capacity_refuses_growth():
    q=PlainLinear(torch.ones(3,2,dtype=torch.float16))
    s=ProjectionScratch([q],2,'cpu')
    with pytest.raises(ValueError,match='rows'):
        matmul(q,torch.ones(3,2),torch.empty(3,3),s)


def test_resident_filter_and_exact_shapes():
    cfg=SimpleNamespace(layers=45,experts=288)
    f=resident_transform(cfg,load_mtp=False)
    info={'shape':[256,1536,64],'dtype':'I16'}
    # Full intact qkv payload, not one rank's half.
    assert f('model.language_model.layers.0.self_attn.qkv_proj.trellis',info)==(256*1536*64*2 + 8*(24576//128)*4,0)
    assert f('model.language_model.layers.45.eh_proj.trellis',info)==(0,0)
    assert f('model.visual.blocks.0.attn.qkv.trellis',info)==(0,0)
    assert f('model.language_model.layers.0.self_attn.f_a_proj.weight',{'shape':[128,4096],'dtype':'F16'})==(128*4096*2,0)
    assert f('lm_head.mul1',{'shape':[],'dtype':'I32'})==(0,0)


def test_composed_rejects_mixed_input_width():
    with pytest.raises(ValueError,match='input'):
        ComposedLinear([PlainLinear(torch.ones(2,3)),PlainLinear(torch.ones(2,4))])


def test_intact_packed_qkv_call_boundary(monkeypatch):
    # Interface-only spy stops BEFORE native arithmetic: no simulated CUDA result.
    tr=torch.zeros(8,24,64,dtype=torch.int16)
    q=Exl3Linear.from_tensors(tr,torch.ones(128,dtype=torch.float16),
                             torch.ones(384,dtype=torch.float16),'mul1',device='cpu',layout='stored')
    composed=ComposedLinear([q,PlainLinear(torch.ones(64,128,dtype=torch.float16))])
    s=ProjectionScratch([composed],3,'cpu')
    class BoundaryReached(Exception): pass
    def stop(self,x,**kw):
        assert self is q and self.words.data_ptr()==tr.data_ptr()
        assert (self.k,self.n)==(128,384)
        assert x.shape==(3,128) and x.is_contiguous()
        assert kw['out'].shape==(3,384) and kw['out'].is_contiguous()
        assert kw['xh'].shape==(3,128) and kw['z'] is s.z
        raise BoundaryReached
    monkeypatch.setattr(Exl3Linear,'__call__',stop)
    with pytest.raises(BoundaryReached):
        matmul(composed,torch.ones(3,128),torch.empty(3,448),s)


def test_routed_native_activation_boundary(monkeypatch):
    from tensorfold.families.glm5_next.cuda.prefill import routed_micro_windows
    from tensorfold.cuda.exl3 import experts
    class BoundaryReached(Exception): pass
    def stop(x,pick,wts,ex,scratch,out,R,**kw):
        assert R==2 and wts is None and out is None
        assert kw['limit']==10 and kw['act_mode']==experts.ACT_BF16
        raise BoundaryReached
    monkeypatch.setattr(experts,'routed',stop)
    with pytest.raises(BoundaryReached):
        routed_micro_windows(torch.ones(2,128),torch.zeros(2,2,dtype=torch.int32),
                             object(),SimpleNamespace(rows=2),torch.empty(2,2,128),2,10)


def test_mtp_omission_fails_before_device_work():
    from tensorfold.families.glm5_next.cuda.mtp import mtp_stage
    with pytest.raises(ValueError,match='omitted'):
        mtp_stage(SimpleNamespace(mtp=None),None,None,[1],torch.ones(1,2))
