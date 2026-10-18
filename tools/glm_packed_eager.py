"""Authorized-GPU eager full-target harness; CPU --help is safe.

Not executed on CUDA by the adapter workstream. Admission precedes loading;
keeps all 45 layers/64 heads/full vocabulary, fresh state, no MTP or drafter.
"""
from __future__ import annotations
import argparse
import json
from dataclasses import replace
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('model_dir',type=Path)
    ap.add_argument('--tokens',required=True,help='JSON file containing immutable raw input token IDs')
    ap.add_argument('--count',type=int,default=16)
    ap.add_argument('--prefill-rows',type=int,default=128)
    ap.add_argument('--receipt',type=Path,required=True)
    ap.add_argument('--authorize-gpu-load',action='store_true',help='required explicit authorization for full CUDA load')
    a=ap.parse_args()
    if not a.authorize_gpu_load:
        ap.error('full CUDA load requires --authorize-gpu-load')
    if a.count<1 or not 1<=a.prefill_rows<=2048:
        ap.error('count must be positive and prefill rows in 1..2048')
    tokens=json.loads(Path(a.tokens).read_text())
    import torch
    from tensorfold.cuda.capacity import admit, headers, _meminfo, GIB
    from tensorfold.cuda.geometry import mla_geometry, with_fixed
    from tensorfold.families.glm5_next.cuda.weights import Config, load, packed_residency, workspace_bytes
    from tensorfold.families.glm5_next.cuda.decode import Engine, prefill, serial_decode
    from tensorfold.families.glm5_next.cuda import LATENT
    cfg=replace(Config.read(a.model_dir),mtp_layers=0)
    if (cfg.layers,cfg.heads,cfg.lin_heads,cfg.vocab)!=(45,64,64,154880) or not LATENT:
        ap.error('harness requires complete GLM-5.3 Flash geometry and latent cache')
    if not isinstance(tokens,list) or not tokens or any(type(v) is not int or not 0<=v<cfg.vocab for v in tokens):
        ap.error('tokens must be nonempty valid integer IDs')
    if len(tokens)+a.count>=cfg.dense_limit:
        ap.error('initial eager harness must stay strictly below dense-to-sparse transition')
    # Existing admission reserves max(4 GiB, 10% host). Add the difference
    # rather than claiming a nonexistent reserve environment variable.
    memory=_meminfo()
    if memory is None:
        raise ValueError('cannot establish required host-memory reserve')
    additional_reserve=max(0,20*GIB-max(4*GIB,memory['MemTotal']//10))
    runtime_allowance=3*GIB # planning allowance, NOT a measured runtime bound
    extra=workspace_bytes(cfg,headers(a.model_dir),rows=1,prefill_rows=a.prefill_rows,load_mtp=False)
    def geometry(t):
        effective={**t,'num_nextn_predict_layers':0}
        return with_fixed(mla_geometry(effective,1,1,minimum_slots=2560,latent=True),extra+additional_reserve+runtime_allowance)
    admission=admit(a.model_dir,len(tokens)+a.count,True,torch,geometry,packed_residency(cfg,load_mtp=False),rank=0,world=1)
    w=load(a.model_dir,rank=0,world=1,load_mtp=False)
    if len(w.layers)!=45 or not w.meta.get('packed_exl3') or w.mtp is not None:
        raise ValueError('loader did not preserve the complete packed target-only contract')
    e=Engine(w,capacity=admission['cache_slots'],max_rows=1,prefill_rows=a.prefill_rows,graphs=False,long_context=False)
    first=prefill(e,tokens,None,mtp=False)
    result=serial_decode(e,first,a.count,None,stop_eos=False)
    receipt=dict(input_tokens=tokens,output_tokens=result.tokens,seconds=result.seconds,
                 layers=len(w.layers),heads=cfg.heads,vocab=w.head.n,world=w.world,
                 mtp_loaded=w.mtp is not None,graphs=False,replays=e.replays,
                 resident_storage_bytes=w.nbytes(),admission=admission,
                 adapter_workspace_bytes=extra,additional_host_reserve_bytes=additional_reserve,
                 runtime_allowance_bytes=runtime_allowance,torch_version=torch.__version__)
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    a.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
