"""Packed GLM prompt helpers: bounded native micro-windows, honest plain dtype.

Plain matmul accumulates FP32 products without converting stored F16 to BF16.
Fixed per-row arithmetic; CUDA/native numerical qualification remains separate.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _plain(X,W,Y,M,xs,ys,N:tl.constexpr,K:tl.constexpr,BK:tl.constexpr,BN:tl.constexpr):
    r=tl.program_id(0)
    n=tl.program_id(1)*BN+tl.arange(0,BN)
    k=tl.arange(0,BK)
    acc=tl.zeros((BN,),tl.float32)
    for k0 in range(0,tl.cdiv(K,BK)):
        kk=k0*BK+k
        x=tl.load(X+r*xs+kk,mask=kk<K,other=0).to(tl.float32)
        w=tl.load(W+n[:,None]*K+kk[None,:],mask=(n[:,None]<N)&(kk[None,:]<K),other=0).to(tl.float32)
        acc=acc+tl.sum(w*x[None,:],axis=1)
    tl.store(Y+r*ys+n,acc,mask=n<N)


def plain_matmul(weight,x,out):
    m,k=x.shape
    n=weight.shape[0]
    if weight.shape[1]!=k or out.shape!=(m,n) or not weight.is_contiguous():
        raise ValueError('plain matmul shape/contiguity mismatch')
    _plain[(m,triton.cdiv(n,16))](x,weight,out,m,x.stride(0),out.stride(0),N=n,K=k,BK=128,BN=16,num_warps=4)
    return out


def routed_micro_windows(x,pick,ex,scratch,out,rows,limit):
    """Native generic compressed routed experts, copied per-slot before combine.

    Scratch slots include the shared slot (pick==E); native leaves that slot to
    caller. Shared output is installed afterward by the normal GLM combine.
    """
    from tensorfold.cuda.exl3.experts import routed, ACT_BF16
    for start in range(0,rows,scratch.rows):
        r=min(scratch.rows,rows-start)
        y=routed(x[start:start+r],pick[start:start+r],None,ex,scratch,None,r,
                 limit=limit,act_mode=ACT_BF16)
        out[start:start+r].copy_(y.view(r,scratch.slots,ex.dims))
    return out
