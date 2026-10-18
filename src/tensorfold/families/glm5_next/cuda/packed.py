"""GLM packed-module composition around unchanged upstream generic EXL3.

Packed matrices are NEVER sliced or concatenated. Decode uses <=128-row
micro-windows for prompts too: bounded scratch, no decoded-W residency, same
native dense arithmetic for serial and prompt rows. All calls run on one stream.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import re
import torch
from tensorfold.cuda.exl3.linear import Exl3Linear


@dataclass
class PlainLinear:
    weight: torch.Tensor

    def __post_init__(self):
        if self.weight.ndim != 2 or self.weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError('plain projection needs a floating [output,input] matrix')
        self.weight = self.weight.contiguous()
        self.n, self.k = self.weight.shape


@dataclass
class PlainAbsorb:
    """Stored-dtype head-major latent K/V blocks; expanded linears alias them."""
    wk: torch.Tensor
    wv: torch.Tensor

    def __post_init__(self):
        self.heads, self.qk_dim, self.lw = self.wk.shape
        self.v_dim = self.wv.shape[1]
        if self.wv.shape[0] != self.heads or self.wv.shape[2] != self.lw:
            raise ValueError("latent K/V head shapes disagree")


class ComposedLinear:
    def __init__(self, parts):
        self.parts = tuple(parts)
        if not self.parts or len({q.k for q in self.parts}) != 1:
            raise ValueError('composed projections need the same input width')
        self.k = self.parts[0].k
        offsets = [0]
        for q in self.parts:
            offsets.append(offsets[-1] + q.n)
        self.offsets = tuple(offsets)
        self.n = offsets[-1]


def leaves(q):
    if isinstance(q, ComposedLinear):
        for p in q.parts:
            yield from leaves(p)
    else:
        yield q


class ProjectionScratch:
    """Fixed capacity, no lazy allocations in CUDA matmul; shared sequentially.

    Component output is contiguous for the native C++ ABI. Copying to composed
    output slices happens AFTER its output rotation/rounding, never before.
    """
    def __init__(self, projections, rows, device, prefill=False):
        if rows < 1:
            raise ValueError('scratch rows must be positive')
        mats = [p for q in projections for p in leaves(q)]
        self.rows, self.chunk = rows, min(rows, 128)
        self.k = max((q.k for q in mats), default=1)
        self.n = max((q.n for q in mats), default=1)
        self.xh = torch.empty(self.chunk*self.k,dtype=torch.float16,device=device)
        self.inputs = {dt:torch.empty(self.chunk*self.k,dtype=dt,device=device)
                       for dt in (torch.bfloat16,torch.float16,torch.float32)}
        self.outputs = {dt:torch.empty(self.chunk*self.n,dtype=dt,device=device)
                        for dt in (torch.bfloat16,torch.float16,torch.float32)}
        z = max((q.split[0]*q.n for q in mats if isinstance(q,Exl3Linear) and q.split[0]>1),default=1)
        self.z = torch.empty(self.chunk*z,dtype=torch.float32,device=device)
        # Trigger native lazy counter creation BEFORE any capture.
        for q in mats:
            if isinstance(q,Exl3Linear):
                _ = q.counters

    def nbytes(self):
        return sum(t.numel()*t.element_size() for t in
                   [self.xh,self.z,*self.inputs.values(),*self.outputs.values()])


def _plain_cuda(q,x,out):
    # Import is lazy: CPU contracts never compile or load CUDA extensions.
    from .prefill import plain_matmul
    return plain_matmul(q.weight,x,out)


def matmul(q, x, out, scratch):
    m = x.shape[0]
    if m > scratch.rows:
        raise ValueError(f'{m} rows exceed scratch capacity {scratch.rows}')
    if x.ndim != 2 or x.shape[1] != q.k or out.numel()!=m*q.n or out.stride(-1)!=1:
        raise ValueError('projection input/output shape mismatch')
    dst = out.view(m,q.n)
    if isinstance(q,ComposedLinear):
        for j,p in enumerate(q.parts):
            matmul(p,x,dst[:,q.offsets[j]:q.offsets[j+1]],scratch)
        return out
    if q.k>scratch.k or q.n>scratch.n:
        raise ValueError('projection exceeds preallocated scratch dimensions')
    for start in range(0,m,scratch.chunk):
        r=min(scratch.chunk,m-start)
        xi=scratch.inputs[x.dtype][:r*q.k].view(r,q.k)
        xi.copy_(x[start:start+r])
        yi=scratch.outputs[out.dtype][:r*q.n].view(r,q.n)
        if isinstance(q,Exl3Linear):
            q(xi,out=yi,xh=scratch.xh[:r*q.k].view(r,q.k),z=scratch.z)
        elif isinstance(q,PlainLinear):
            if x.device.type=='cpu':
                # Real CPU plain contract, NOT a packed/CUDA arithmetic oracle.
                yi.copy_(xi.float() @ q.weight.float().T)
            else:
                _plain_cuda(q,xi,yi)
        else:
            raise TypeError(f'unsupported packed adapter projection {type(q)}')
        dst[start:start+r].copy_(yi)
    return out


def projections(w):
    """All modules used by target and optional MTP, no duplicate head."""
    yield w.head
    layers=list(w.layers)
    if w.mtp is not None:
        yield w.mtp.eh
        layers.append(w.mtp.layer)
    for l in layers:
        if l.kda is not None:
            for k in ('proj','fb','gb','o'):
                yield getattr(l.kda,k)
        if l.dsa is not None:
            for k in ('proj','q_b','o','kv_k','kv_v'):
                q=getattr(l.dsa,k)
                if q is not None: yield q
            if l.dsa.index is not None:
                yield l.dsa.index.kw
                yield l.dsa.index.qb
                yield PlainLinear(l.dsa.index.gate)
        m=l.mlp if l.mlp is not None else l.moe.shared
        if m is not None:
            yield m.gu
            yield m.down
        if l.moe is not None:
            yield PlainLinear(l.moe.router)


def resident_transform(cfg, *, load_mtp=True):
    """Exact tensor payload residency plus native pointer/counter tables, TP1.

    Excludes discarded codebook marker scalars and unused vision/MTP. Includes
    no graph pool, allocator padding, or scratch; caller budgets these separately.
    No permanent decoded matrix, duplicated head or latent BF16 copies.
    """
    sizes={'I16':2,'F16':2,'BF16':2,'F32':4,'I32':4,'I64':8,'U8':1}
    def transform(name,info):
        if name.startswith('model.visual.'):
            return 0,0
        m=re.search(r'\.layers\.(\d+)\.',name)
        if m and int(m[1])>=cfg.layers and not load_mtp:
            return 0,0
        if name.endswith(('.mul1','.mcg','.3inst')):
            return 0,0
        if not (name.startswith('model.language_model.') or name.startswith('lm_head.')):
            raise ValueError(f'uninventoried packed GLM tensor {name}')
        amount=math.prod(info['shape'])*sizes[info['dtype']]
        if name.endswith('.trellis'):
            if '.experts.' in name:
                amount+=12  # one device pointer and K2 per projection
                if '.down_proj.' in name: amount+=8 # CPU trellis-byte table
            else:
                amount+=8*(info['shape'][1]*16//128)*4 # initialized native counters
        return amount,0
    return transform


def fp32_residual(w):
    """Native-width mHC carrier is confined to the intact packed TP1 adapter."""
    return w.world == 1 and bool(w.meta.get("packed_exl3", False))


def workspace_bytes(cfg, headers, *, rows=8, prefill_rows=2048, load_mtp=True):
    """Exact adapter scratch shapes (not total runtime/allocator high-water).

    Mirrors ProjectionScratch and generic Scratch; conservative plain matrices
    may include loader-only matrices. Admission adds this to existing geometry,
    which already counts legacy buffers and cache. No decoded-W buffer exists.
    """
    from tensorfold.cuda.exl3.linear import plan
    from tensorfold.cuda.exl3.experts import default_config
    filt = resident_transform(cfg, load_mtp=load_mtp)
    kmax = nmax = zmax = 1
    for name, info in headers.items():
        if not filt(name, info)[0] or '.experts.' in name:
            continue
        sh = info['shape']
        if name.endswith('.trellis'):
            k,n = sh[0]*16,sh[1]*16
            sk,_ = plan(k,n)
            if sk>1: zmax=max(zmax,sk*n)
        elif name.endswith('.weight') and len(sh)==2:
            n,k = sh
        else:
            continue
        kmax,nmax=max(kmax,k),max(nmax,n)
    D,I,E=cfg.hidden,cfg.moe_width,cfg.experts
    slots=cfg.top_k+1
    gu=default_config(D,I,True)
    down=default_config(I,D,False)
    def one(r):
        r=min(r,128)
        projection=r*(2*kmax+8*kmax+8*nmax+4*zmax)
        P=r*slots
        U=min(P,E)
        routed=P*(4*D+2*I+4*max(2*gu[2]*I,down[2]*D)+4*D)+4*U+4+4*U*r
        return projection+routed
    mtp_rows = rows if load_mtp and cfg.mtp_layers else 0
    # Existing geometry budgets BF16 x. Every Buffers instance now has FP32 x,
    # including the otherwise unused MTP x. Carrier rows are NOT microchunked.
    carrier_delta = (rows + prefill_rows + mtp_rows) * cfg.streams * D * 2
    return one(rows)+one(prefill_rows)+(one(rows) if mtp_rows else 0)+carrier_delta
