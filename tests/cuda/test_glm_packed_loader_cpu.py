"""Real safetensors->RankReader->packed loader CPU fixture contracts.

Small synthetic dense layers exercise actual loader/format constructors, not
full GLM/CUDA inference. Invalid unused MTP/vision tensors prove omitted reads.
"""
import json
import torch
import pytest
from safetensors.torch import save_file
from tensorfold.families.glm5_next.cuda import weights
from tensorfold.families.glm5_next.cuda.packed import ComposedLinear, PlainLinear
from tensorfold.cuda.exl3.linear import Exl3Linear


@pytest.fixture
def checkpoint(tmp_path):
    t=dict(hidden_size=128,num_hidden_layers=2,vocab_size=256,rms_norm_eps=1e-5,
           num_attention_heads=1,q_lora_rank=128,kv_lora_rank=128,
           qk_nope_head_dim=128,qk_rope_head_dim=0,v_head_dim=128,
           linear_attn_config=dict(num_heads=1,head_dim=128,short_conv_kernel_size=4),
           n_routed_experts=2,num_experts_per_tok=1,moe_intermediate_size=128,n_shared_experts=1,
           intermediate_size=128,routed_scaling_factor=2.5,hc_mult=4,
           layer_types=['linear_attention','deepseek_sparse_attention'],
           mlp_layer_types=['dense','dense'],eos_token_id=[255],num_nextn_predict_layers=1,
           index_n_heads=2,index_head_dim=128,index_topk=16,index_kpool=4,swiglu_limit=10.)
    (tmp_path/'config.json').write_text(json.dumps(dict(text_config=t,quantization_config=dict(quant_method='exl3',bits=2.05,codebook='mul1'))))
    tensors={}
    def plain(n,shape,dtype=torch.float16):
        tensors[n]=torch.arange(__import__('math').prod(shape),dtype=torch.float32).reshape(shape).remainder(13).add_(.125).to(dtype)
    def quant(n,k,out,bits):
        tensors[n+'.trellis']=torch.arange(k//16*out//16*16*bits,dtype=torch.int16).reshape(k//16,out//16,16*bits)
        tensors[n+'.suh']=torch.ones(k,dtype=torch.float16)
        tensors[n+'.svh']=torch.ones(out,dtype=torch.float16)
        tensors[n+'.mul1']=torch.tensor(1,dtype=torch.int32)
    P=weights.PREFIX
    plain(P+'embed_tokens.weight',[256,128],torch.bfloat16)
    plain(P+'norm.weight',[128],torch.bfloat16)
    quant('lm_head',128,256,5)
    for i in range(2):
        p=P+f'layers.{i}.'
        for norm in ('input_layernorm','post_attention_layernorm'):
            plain(p+norm+'.weight',[128],torch.bfloat16)
        for site in ('attn','ffn'):
            plain(p+f'hc_{site}_fn',[24,512],torch.float32)
            plain(p+f'hc_{site}_base',[24],torch.float32)
            plain(p+f'hc_{site}_scale',[3],torch.float32)
        for proj in ('gate','up','down'):
            quant(p+f'mlp.{proj}_proj',128,128,3)
        a=p+'self_attn.'
        if i==0:
            quant(a+'qkv_proj',128,384,4)
            for proj,out,k in [('f_a',128,128),('g_a',128,128),('b',1,128),('f_b',128,128),('g_b',128,128)]:
                plain(a+proj+'_proj.weight',[out,k])
            plain(a+'conv1d.weight',[384,1,4],torch.bfloat16)
            plain(a+'A_log',[1],torch.float32)
            plain(a+'dt_bias',[128],torch.float32)
            plain(a+'o_norm.weight',[128],torch.bfloat16)
            quant(a+'o_proj',128,128,4)
        else:
            for proj,k,out in [('q_a_proj',128,128),('kv_a_proj_with_mqa',128,128),('q_b_proj',128,128),('o_proj',128,128),('indexer.wq_b',128,256)]:
                quant(a+proj,k,out,4)
            plain(a+'kv_b_proj.weight',[256,128])
            for norm in ('q_a_layernorm','kv_a_layernorm'):
                plain(a+norm+'.weight',[128],torch.bfloat16)
            plain(a+'indexer.wk.weight',[128,128])
            plain(a+'indexer.weights_proj.weight',[2,128])
            plain(a+'indexer.k_norm.weight',[128])
            plain(a+'indexer.k_norm.bias',[128])
            plain(a+'indexer.index_kpool_compress_gate',[128,128])
            plain(a+'indexer.index_kpool_compress_ape',[4,128],torch.float32)
    # Deliberately invalid: omission must be governed by load contract, not dtype.
    tensors[P+'layers.2.eh_proj.trellis']=torch.tensor([42],dtype=torch.int16)
    tensors['model.visual.invalid']=torch.tensor([99])
    save_file(tensors,str(tmp_path/'model.safetensors'))
    return tmp_path,tensors


def test_actual_target_only_loader(checkpoint):
    root,t=checkpoint
    w=weights.load(root,rank=0,world=1,load_mtp=False,device='cpu')
    assert w.cfg.bits==2.05 and w.cfg.mtp_layers==0
    assert w.mtp is None and w.draft_head is None
    assert w.world==1 and w.vocab_offset==0 and w.head.n==256 and w.head.bits==5
    assert len(w.layers)==2 and w.layers[0].kda.heads==1
    k=w.layers[0].kda
    assert isinstance(k.proj,ComposedLinear) and k.proj.offsets==(0,384,512,640,641)
    assert isinstance(k.proj.parts[0],Exl3Linear) and k.proj.parts[0].bits==4
    assert torch.equal(k.proj.parts[0].words.view(torch.int16),t[weights.PREFIX+'layers.0.self_attn.qkv_proj.trellis'])
    assert isinstance(k.fb,PlainLinear) and k.fb.weight.dtype==torch.float16
    assert torch.equal(w.norm,t[weights.PREFIX+'norm.weight']) # plain gamma, no centering
    assert w.layers[0].mlp.gu.parts[0].bits==3
    d=w.layers[1].dsa
    assert d.absorb.wk.dtype==torch.float16
    assert d.absorb.wk.data_ptr()==d.kv_k.weight.data_ptr()
    assert d.absorb.wv.data_ptr()==d.kv_v.weight.data_ptr()
    assert torch.equal(d.absorb.wk,t[weights.PREFIX+'layers.1.self_attn.kv_b_proj.weight'][:128].view(1,128,128))
    # Actual storage traversal includes aliasing and generic native objects.
    assert w.nbytes()>0


def test_packed_tp2_rejected_before_tensor_reads(checkpoint):
    root,_=checkpoint
    with pytest.raises(ValueError,match='intact packed GLM'):
        weights.load(root,rank=0,world=2,load_mtp=False,device='cpu')


def test_unused_mtp_not_silently_enabled(checkpoint):
    root,_=checkpoint
    with pytest.raises((KeyError,ValueError)):
        weights.load(root,rank=0,world=1,load_mtp=True,device='cpu')


def test_actual_buffer_allocation_target_only(checkpoint):
    from tensorfold.families.glm5_next.cuda.forward import Buffers, State
    from tensorfold.families.glm5_next.cuda.packed import matmul
    root,_=checkpoint
    w=weights.load(root,rank=0,world=1,load_mtp=False,device='cpu')
    b=Buffers(w,260,capacity=32,prefill=True)
    st=State(w,32,3)
    assert b.hin is b.me is b.mx is b.mcat is b.mxs is None
    assert not hasattr(st,'mtp_kc')
    assert st.rec.shape[:3]==(2,1,1) # recurrent parity remains two
    assert b.projection_scratch.chunk==128 and b.sk.numel()==1
    s=b.projection_scratch
    pointers=[t.data_ptr() for t in [s.xh,s.z,*s.inputs.values(),*s.outputs.values()]]
    q=w.layers[0].kda.fb
    x=torch.ones(260,q.k,dtype=torch.bfloat16)
    out=torch.empty(260,q.n,dtype=torch.bfloat16)
    matmul(q,x,out,s)
    expected=(x.float()@q.weight.float().T).bfloat16()
    torch.testing.assert_close(out,expected,rtol=0,atol=0)
    assert pointers==[t.data_ptr() for t in [s.xh,s.z,*s.inputs.values(),*s.outputs.values()]]
