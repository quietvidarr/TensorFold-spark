"""Read all real GLM headers only; no tensor payload/GPU/extension access."""
import argparse
import json
from collections import Counter
from tensorfold.cuda.capacity import headers
from tensorfold.families.glm5_next.cuda.weights import Config, packed_residency, workspace_bytes, PREFIX


def audit(root, load_mtp=False):
    c=Config.read(root)
    h=headers(root)
    used=set()
    widths=Counter()
    def plain(name,shape=None,dtype=None):
        e=h[name]
        if shape is not None and e['shape']!=list(shape): raise ValueError(f'{name}: shape {e["shape"]} != {shape}')
        if dtype is not None and e['dtype']!=dtype: raise ValueError(f'{name}: dtype {e["dtype"]} != {dtype}')
        used.add(name)
    def group(name,k,n,bits):
        plain(name+'.trellis',(k//16,n//16,16*bits),'I16')
        plain(name+'.suh',(k,),'F16')
        plain(name+'.svh',(n,),'F16')
        plain(name+'.mul1',(),'I32')
        widths[bits]+=1
    def proj(name,k,n,bits):
        if name+'.trellis' in h: group(name,k,n,bits)
        else: plain(name+'.weight',(n,k))
    D=c.hidden
    plain(PREFIX+'embed_tokens.weight',(c.vocab,D),'BF16')
    plain(PREFIX+'norm.weight',(D,),'BF16')
    group('lm_head',D,c.vocab,5)
    for i in range(c.layers+int(load_mtp and c.mtp_layers>0)):
        mtp=i==c.layers
        p=PREFIX+f'layers.{i}.'
        for norm in ('input_layernorm','post_attention_layernorm'):
            plain(p+norm+'.weight',(D,),'BF16')
        if mtp:
            for norm in ('enorm','hnorm','shared_head.norm'): plain(p+norm+'.weight',(D,),'BF16')
            group(p+'eh_proj',2*D,D,2)
        else:
            for site in ('attn','ffn'):
                plain(p+f'hc_{site}_fn',(24,c.streams*D),'F32')
                plain(p+f'hc_{site}_base',(24,),'F32')
                plain(p+f'hc_{site}_scale',(3,),'F32')
        a=p+'self_attn.'
        if not mtp and c.kinds[i]=='kda':
            width=c.lin_heads*c.lin_dim
            group(a+'qkv_proj',D,3*width,4)
            plain(a+'conv1d.weight',(3*width,1,c.conv),'BF16')
            for n,k,out in [('f_a',D,128),('g_a',D,128),('b',D,c.lin_heads),('f_b',128,width),('g_b',128,width)]:
                plain(a+n+'_proj.weight',(out,k),'F16')
            plain(a+'A_log',(c.lin_heads,),'F32')
            plain(a+'dt_bias',(width,),'F32')
            plain(a+'o_norm.weight',(128,),'BF16')
            group(a+'o_proj',width,D,4)
        else:
            for n,k,out in [('q_a_proj',D,c.q_lora),('kv_a_proj_with_mqa',D,c.kv_lora),('q_b_proj',c.q_lora,c.heads*c.qk_dim),('o_proj',c.heads*c.v_dim,D),('indexer.wq_b',c.q_lora,c.index_heads*c.index_dim)]:
                group(a+n,k,out,4)
            plain(a+'kv_b_proj.weight',(c.heads*(c.qk_dim+c.v_dim),c.kv_lora),'F16')
            plain(a+'q_a_layernorm.weight',(c.q_lora,),'BF16')
            plain(a+'kv_a_layernorm.weight',(c.kv_lora,),'BF16')
            for n,out in [('wk',c.index_dim),('weights_proj',c.index_heads)]:
                plain(a+'indexer.'+n+'.weight',(out,D),'F16')
            for suffix in ('weight','bias'): plain(a+'indexer.k_norm.'+suffix,(c.index_dim,),'F16')
            plain(a+'indexer.index_kpool_compress_gate',(c.index_dim,D),'F16')
            plain(a+'indexer.index_kpool_compress_ape',(c.kpool,c.index_dim),'F32')
        m=p+'mlp.'
        if not mtp and c.mlp_kinds[i]=='dense':
            for n,k,out in [('gate_proj',D,c.dense_width),('up_proj',D,c.dense_width),('down_proj',c.dense_width,D)]: group(m+n,k,out,3)
        else:
            plain(m+'gate.weight',(c.experts,D),'F16')
            plain(m+'gate.e_score_correction_bias',(c.experts,),'F32')
            for e in range(c.experts):
                for n,k,out in [('gate_proj',D,c.moe_width),('up_proj',D,c.moe_width),('down_proj',c.moe_width,D)]: group(m+f'experts.{e}.'+n,k,out,2)
            for n,k,out in [('gate_proj',D,c.shared_width),('up_proj',D,c.shared_width),('down_proj',c.shared_width,D)]: group(m+'shared_experts.'+n,k,out,4)
    f=packed_residency(c,load_mtp=load_mtp)
    resident=sum(f(n,e)[0] for n,e in h.items())
    serialized=sum(e['data_offsets'][1]-e['data_offsets'][0] for n,e in h.items() if n in used)
    declared={n for n,e in h.items() if f(n,e)[0] or (n.endswith('.mul1') and n in used)}
    extra=declared-used
    if extra: raise ValueError(f'unused admitted tensors: {sorted(extra)[:16]}')
    return dict(header_tensors=len(h),consumed_tensors=len(used),groups_by_bits=dict(widths),layers=c.layers,
                heads=c.heads,lin_heads=c.lin_heads,vocab=c.vocab,load_mtp=load_mtp,
                serialized_selected_bytes=serialized,resident_payload_and_tables_bytes=resident,
                adapter_workspace_rows8_prefill2048_bytes=workspace_bytes(c,h,rows=8,prefill_rows=2048,load_mtp=load_mtp),
                unused_admitted_tensors=len(extra),tensor_payload_read_bytes=0,cuda_executed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model_dir')
    p.add_argument('--load-mtp',action='store_true')
    a=p.parse_args()
    print(json.dumps(audit(a.model_dir,a.load_mtp),indent=2))
