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


@triton.jit
def _plain_rows(X,W,Y,M,xs,ys,N:tl.constexpr,K:tl.constexpr,BK:tl.constexpr,BN:tl.constexpr,BM:tl.constexpr):
    """BM rows a program, each W tile read once for all of them; a row's sum is _plain's (the same per-row tile sums
    in the same order), so its bits never depend on the window or on the other rows."""
    r0=tl.program_id(0)*BM
    n=tl.program_id(1)*BN+tl.arange(0,BN)
    k=tl.arange(0,BK)
    rows=tl.arange(0,BM)
    acc=tl.zeros((BM,BN),tl.float32)
    for k0 in range(0,tl.cdiv(K,BK)):
        kk=k0*BK+k
        w=tl.load(W+n[:,None]*K+kk[None,:],mask=(n[:,None]<N)&(kk[None,:]<K),other=0).to(tl.float32)
        for i in tl.static_range(BM):
            x=tl.load(X+(r0+i)*xs+kk,mask=(kk<K)&(r0+i<M),other=0).to(tl.float32)
            part=tl.sum(w*x[None,:],axis=1)
            acc=tl.where(rows[:,None]==i,acc+part[None,:],acc)
    r=r0+rows
    tl.store(Y+r[:,None]*ys+n[None,:],acc,mask=(r[:,None]<M)&(n[None,:]<N))


PLAIN_ROWS = int(__import__("os").environ.get("TF_GLM_PLAIN_ROWS", "0"))   # rows a program (0: upstream per-row)
# TF_GLM_PLAIN_TC=1 (default): plain F16 matrices on the tensor cores in TF32 (lossless here: an F16 weight and a BF16
# input are exact in TF32, every product is exact in FP32, the sums are FP32), with ONE tile shape and K split for a
# matrix whatever the row count, so a row's bits depend on that row alone (resume stays exact)
PLAIN_TC = __import__("os").environ.get("TF_GLM_PLAIN_TC", "1") != "0"
TC_BM, TC_BN, TC_BK = 16, 32, 64


@triton.jit
def _plain_tc(X,W,P,M,xs,N:tl.constexpr,K:tl.constexpr,KS:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,
              BK:tl.constexpr):
    """Program (row block, column block, K split): P[split, rows, cols] = X[rows, split's K] @ W[cols, split's K]^T."""
    r=tl.program_id(0)*BM+tl.arange(0,BM)
    n=tl.program_id(1)*BN+tl.arange(0,BN)
    s=tl.program_id(2)
    k=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    for k0 in range(0,KS//BK):
        kk=s*KS+k0*BK+k
        x=tl.load(X+r[:,None]*xs+kk[None,:],mask=(r[:,None]<M)&(kk[None,:]<K),other=0).to(tl.float32)
        w=tl.load(W+n[None,:]*K+kk[:,None],mask=(n[None,:]<N)&(kk[:,None]<K),other=0).to(tl.float32)
        acc=tl.dot(x,w,acc,input_precision="tf32")
    tl.store(P+(s*M+r[:,None])*N+n[None,:],acc,mask=(r[:,None]<M)&(n[None,:]<N))


@triton.jit
def _plain_sum(P,Y,M,ys,N:tl.constexpr,S:tl.constexpr,BLOCK:tl.constexpr):
    """Y[r, n] = the K splits' partial sums added in split order (a fixed order)."""
    i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    r=i//N
    n=i%N
    acc=tl.zeros((BLOCK,),tl.float32)
    for s in range(S):
        acc+=tl.load(P+s*M*N+i,mask=i<M*N,other=0)
    tl.store(Y+r*ys+n,acc,mask=i<M*N)


@triton.jit
def _plain_part(X,W,P,M,xs,N:tl.constexpr,K:tl.constexpr,KS:tl.constexpr,BK:tl.constexpr,BN:tl.constexpr):
    """_plain over K split s (program axis 2): the same per-row tile sums, partials to P[s, row, cols]."""
    r=tl.program_id(0)
    n=tl.program_id(1)*BN+tl.arange(0,BN)
    s=tl.program_id(2)
    k=tl.arange(0,BK)
    acc=tl.zeros((BN,),tl.float32)
    for k0 in range(0,KS//BK):
        kk=s*KS+k0*BK+k
        x=tl.load(X+r*xs+kk,mask=kk<K,other=0).to(tl.float32)
        w=tl.load(W+n[:,None]*K+kk[None,:],mask=(n[:,None]<N)&(kk[None,:]<K),other=0).to(tl.float32)
        acc=acc+tl.sum(w*x[None,:],axis=1)
    tl.store(P+(s*M+r)*N+n,acc,mask=n<N)


def decode_splits(n, k):
    """K splits for a decode window's rows (the matrix's shape alone, so a row's bits never depend on R)."""
    blocks = triton.cdiv(n, 16)
    s = 1
    while blocks * s < 256 and k % (128 * s * 2) == 0:
        s *= 2
    return s


def plain_decode(weight, x, out):
    m, k = x.shape
    n = weight.shape[0]
    s = decode_splits(n, k)
    if s == 1:
        _plain[(m,triton.cdiv(n,16))](x,weight,out,m,x.stride(0),out.stride(0),N=n,K=k,BK=128,BN=16,num_warps=4)
        return out
    part = torch.empty((s, m, n), dtype=torch.float32, device=x.device)
    _plain_part[(m, triton.cdiv(n, 16), s)](x, weight, part, m, x.stride(0), N=n, K=k, KS=k // s, BK=128, BN=16,
                                             num_warps=4)
    _plain_sum[(triton.cdiv(m * n, 1024),)](part, out, m, out.stride(0), N=n, S=s, BLOCK=1024, num_warps=4)
    return out


def plain_splits(n, k):
    """K splits for a matrix (its shape alone): enough programs to stream W at full bandwidth for a few rows."""
    blocks = triton.cdiv(n, TC_BN)
    s = 1
    while blocks * s < 128 and k % (TC_BK * s * 2) == 0 and k // (s * 2) >= 256:
        s *= 2
    return s


def plain_tc(weight, x, out):
    m, k = x.shape
    n = weight.shape[0]
    s = plain_splits(n, k)
    grid = (triton.cdiv(m, TC_BM), triton.cdiv(n, TC_BN), s)
    if s == 1 and out.stride(1) == 1 and out.stride(0) == n:
        _plain_tc[grid](x, weight, out, m, x.stride(0), N=n, K=k, KS=k, BM=TC_BM, BN=TC_BN, BK=TC_BK, num_warps=4)
        return out
    part = torch.empty((s, m, n), dtype=torch.float32, device=x.device)
    _plain_tc[grid](x, weight, part, m, x.stride(0), N=n, K=k, KS=k // s, BM=TC_BM, BN=TC_BN, BK=TC_BK, num_warps=4)
    _plain_sum[(triton.cdiv(m * n, 1024),)](part, out, m, out.stride(0), N=n, S=s, BLOCK=1024, num_warps=4)
    return out


PLAIN_DECODE_SPLIT = __import__("os").environ.get("TF_GLM_PLAIN_SPLIT", "1") != "0"


def plain_matmul(weight,x,out,prompt=False):
    m,k=x.shape
    n=weight.shape[0]
    if weight.shape[1]!=k or out.shape!=(m,n) or not weight.is_contiguous():
        raise ValueError('plain matmul shape/contiguity mismatch')
    if prompt and PLAIN_TC and k % TC_BK == 0 and x.stride(1) == 1:
        return plain_tc(weight, x, out)               # prompt rows: tensor cores, one config a matrix
    if not prompt and PLAIN_DECODE_SPLIT and k % 256 == 0 and x.stride(1) == 1:
        return plain_decode(weight, x, out)           # decode windows: per-row sums, K split by the shape alone
    if PLAIN_ROWS > 1:
        bm = PLAIN_ROWS if m >= PLAIN_ROWS else (1 << (m - 1).bit_length())
        _plain_rows[(triton.cdiv(m,bm),triton.cdiv(n,16))](x,weight,out,m,x.stride(0),out.stride(0),N=n,K=k,BK=128,
                                                            BN=16,BM=bm,num_warps=4)
        return out
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
