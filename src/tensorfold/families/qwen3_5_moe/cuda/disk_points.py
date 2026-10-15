"""Qwen3.6 MoE's kept prompt states to and from the prompt store on disk (local patch 0104, ``cuda/prompt_store.py``).

A point is a kept entry of ``MultiDecoder.cache``: the target state (every attention layer's rows below ``pos``, every
DeltaNet layer's recurrent and conv state) and the drafter's (the MTP head's attention rows below its own ``pos`` and
the held hidden rows it has not absorbed). Rows go to the store in blocks of ``B`` (shared by the points of one
conversation); the rows past the last whole block go with the point. A loaded point joins the kept entries, and the
request resumes from it as from any other.
"""

from __future__ import annotations

import torch

from tensorfold.cuda import prompt_store as ps
from tensorfold.families.qwen3_5.cuda.forward import State

from .mtp import Cache

B = ps.B


def _attn(st) -> list:
    return [kv for kv in st.kv if kv is not None]


def _take(st, mc, a: int, b: int, head: bool, cp) -> dict:
    """Rows [a, b) of every attention layer (and of the head's cache), copied to the host by ``cp`` (patch 0106)."""

    layers = _attn(st)
    out = {"k": cp.host(torch.stack([k[a:b] for k, _ in layers]), private=True),
           "v": cp.host(torch.stack([v[a:b] for _, v in layers]), private=True)}
    if head:
        out["hk"], out["hv"] = cp.host(mc.k[a:b]), cp.host(mc.v[a:b])
    return out


def _bytes(obj) -> int:
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(_bytes(v) for v in obj.values())
    if isinstance(obj, list):
        return sum(_bytes(v) for v in obj)
    return 0


def geometry(w) -> str:
    """What the saved rows' layout depends on, beyond the model key: a different one never reads them."""

    st = State(w)
    k = _attn(st)[0][0]
    rec = next(r for r in st.rec if r is not None)
    conv = next(c for c in st.conv if c is not None)
    linear = sum(r is not None for r in st.rec)
    return (f"qwen3_5_moe|bf16|{len(_attn(st))}+{linear}|{tuple(k.shape[1:])}|{tuple(rec.shape)}|"
            f"{tuple(conv.shape)}|B{B}")


def open_store(dec):
    return ps.open_store(geometry(dec.w), idle=lambda: not dec.streams and not dec.filling)


def capture(dec, ids: list[int], state, snap, *, force: bool = False) -> None:
    """Queue a kept entry for the store (its new blocks and the rows past them), unless it is there already. While the
    queue is full (patch 0107) the entry waits in GPU memory (``dec.unsaved``; its rows are views commits never write
    and its recurrent states are copies, so holding it keeps it intact) until ``retry`` finds room."""

    store = dec.store
    n = len(ids)
    mc, held = snap
    if state.pos != n or state.rope_delta or mc.pos > n:
        return
    pre = store.prefix(ids)
    k = pre.point(n)
    if store.has_point(k):
        return
    if not force and store.full():
        if all(u[0] != k for u in dec.unsaved):
            dec.unsaved.append((k, list(ids), state, snap))
            store.defer(k, n)
        return
    cp = store.copier
    full = min((n - 1) // B, mc.pos // B)
    order = [pre.block(j) for j in range(full)]
    blocks = {h: _take(state, mc, j * B, (j + 1) * B, True, cp) for j, h in enumerate(order) if not store.has_block(h)}
    point = {"ids": torch.tensor(ids, dtype=torch.int32), "blocks": order, "pos": n, "head_pos": mc.pos,
             "rows": _take(state, mc, full * B, n, False, cp), "head": {"k": cp.host(mc.k[full * B:mc.pos]),
                                                                       "v": cp.host(mc.v[full * B:mc.pos])},
             "rec": [cp.host(r) for r in state.rec if r is not None],
             "conv": [cp.host(c) for c in state.conv if c is not None], "held": cp.host(held)}
    store.put(k, n, ids, point, blocks, order, _bytes(point) + sum(_bytes(b) for b in blocks.values()),
              ready=cp.done(), forced=force and store.full())


def retry(dec, *, force: bool = False) -> None:
    """Copy the waiting entries, oldest first, while the queue has room (all of them: ``force``)."""

    for u in list(dec.unsaved):
        if not force and dec.store.full():
            return
        dec.unsaved = [x for x in dec.unsaved if x is not u]
        try:
            capture(dec, u[1], u[2], u[3], force=force)
        except Exception as exc:  # noqa: BLE001 - the disk copy never stops a request
            dec.store.stats["errors"] += 1
            print(f"[prompt-cache] save failed: {type(exc).__name__}: {exc}", flush=True)
        if all(x[0] != u[0] for x in dec.unsaved):
            dec.store.deferred.pop(u[0], None)


def restore(dec, prompt: list[int]) -> None:
    """Load the longest stored point this prompt extends into the kept entries, when that beats recomputing from the
    longest kept entry in GPU memory."""

    store = dec.store
    have = max((len(e[0]) for e in dec.cache.entries
                if len(e[0]) < len(prompt) and list(prompt[:len(e[0])]) == e[0]), default=0)
    pre = store.prefix(prompt)
    hit = store.lookup(pre, have, len(prompt))
    if hit is None:
        return
    k, n, nbytes = hit
    try:
        point, blocks, t0 = store.read(k)
        if point["ids"].numel() != n or not torch.equal(point["ids"], torch.tensor(prompt[:n], dtype=torch.int32)):
            raise ValueError("a stored point's tokens differ from its key's")
        dev = dec.w.norm.device
        st = State(dec.w)
        ks = torch.cat([b["k"] for b in blocks] + [point["rows"]["k"]], dim=1).to(dev)
        vs = torch.cat([b["v"] for b in blocks] + [point["rows"]["v"]], dim=1).to(dev)
        if ks.shape[1] != n:
            raise ValueError(f"a stored point holds {ks.shape[1]} rows, not {n}")
        j = 0
        for i, kv in enumerate(st.kv):
            if kv is not None:
                st.kv[i] = (ks[j], vs[j])
                j += 1
        st.rec = [None if r is None else point["rec"].pop(0).to(dev) for r in st.rec]
        st.conv = [None if c is None else point["conv"].pop(0).to(dev) for c in st.conv]
        st.pos = n
        mc = object.__new__(Cache)
        mc.k = torch.cat([b["hk"] for b in blocks] + [point["head"]["k"]]).to(dev)
        mc.v = torch.cat([b["hv"] for b in blocks] + [point["head"]["v"]]).to(dev)
        mc.pos = int(point["head_pos"])
        held = point["held"].to(dev)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        store.loaded(k, n, nbytes, t0)
    except Exception as exc:  # noqa: BLE001 - a point that can't load is recomputed
        store.stats["errors"] += 1
        print(f"[prompt-cache] load failed, prefilling instead: {type(exc).__name__}: {exc}", flush=True)
        return
    dec.cache.add(list(prompt[:n]), st, (mc, held))
