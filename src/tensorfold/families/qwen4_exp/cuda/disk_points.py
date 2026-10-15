"""Flash Next's kept prompt points to and from the prompt store on disk (local patch 0006, ``cuda/prompt_store.py``).

A point is a slot's attention rows below its position (every layer's keys, values, index keys and complete pools, the
MTP head's below its own length) plus the snapshot ``State.restore`` takes (the DeltaNet and conv states, the n-gram
windows) and the MTP tail. Rows go to the store in blocks of ``B`` (shared by the points of one conversation); the rows
past the last whole block go with the point.
"""

from __future__ import annotations

import torch

from tensorfold.cuda import prompt_store as ps

B = ps.B


def _caches(st):
    main = [(st.kc[i], st.ikc[i], st.pooled[i]) for i in range(len(st.kc))]
    mtp = [(st.mtp_kc, st.mtp_ikc, st.mtp_pooled)] if hasattr(st, "mtp_kc") else []
    return main, mtp


def _take(layers, a: int, b: int, ratio: int, cp) -> dict:
    """Rows [a, b) of every layer (``a`` a whole block) and the complete pools among them, copied to the host by ``cp``."""

    if not layers:
        return {}
    out = {"k": torch.stack([c.k[a:b] for c, _, _ in layers]), "v": torch.stack([c.v[a:b] for c, _, _ in layers]),
           "i": torch.stack([i[a:b] for _, i, _ in layers]),
           "p": torch.stack([p[a // ratio:b // ratio] for _, _, p in layers])}
    if layers[0][0].quantized:
        out["ks"] = torch.stack([c.ks[a:b] for c, _, _ in layers])
        out["vs"] = torch.stack([c.vs[a:b] for c, _, _ in layers])
    return {name: cp.host(t, private=True) for name, t in out.items()}   # stacks: fresh tensors


def _put(layers, a: int, d: dict, ratio: int) -> None:
    if not layers:
        return
    b = a + d["k"].shape[1]
    for j, (c, i, p) in enumerate(layers):
        c.k[a:b].copy_(d["k"][j], non_blocking=True)
        c.v[a:b].copy_(d["v"][j], non_blocking=True)
        if c.quantized:
            c.ks[a:b].copy_(d["ks"][j], non_blocking=True)
            c.vs[a:b].copy_(d["vs"][j], non_blocking=True)
        i[a:b].copy_(d["i"][j], non_blocking=True)
        p[a // ratio:a // ratio + d["p"].shape[1]].copy_(d["p"][j], non_blocking=True)


def _bytes(obj) -> int:
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(_bytes(v) for v in obj.values())
    return 0


def geometry(st) -> str:
    """What the saved rows' layout depends on, beyond the model key: a different one never reads them."""

    main, mtp = _caches(st)
    c = main[0][0]
    return (f"qwen4_exp|{st.kv_dtype}|{len(main)}+{len(mtp)}|{c.kv_heads}x{c.head_dim}|{st.index_dim}|{st.ratio}|"
            f"{tuple(st.rec.shape[1:])}|{tuple(st.conv.shape)}|{tuple(st.ple_tail.shape)}|B{B}")


def open_store(dec):
    st = dec.free[0]
    if B % st.ratio:
        print(f"[prompt-cache] off: attention pools of {st.ratio} rows don't divide {B}-row blocks", flush=True)
        return None
    return ps.open_store(geometry(st), idle=lambda: not dec.streams and not dec.filling)


def capture(dec, ids: list[int], st, snap: dict, tail, *, force: bool = False) -> None:
    """Queue a kept point for the store (its new blocks and the rows past them), unless it is there already. While the
    queue is full (patch 0010) the point waits in its GPU slot (``dec.unsaved``) instead: ``retry`` copies it when the
    queue has room, ``flush`` at once when its slot is about to be reused (``force``)."""

    store = dec.store
    n = len(ids)
    if snap.get("pos") != n or st.pos < n or snap.get("image_positions") is not None:
        return
    pre = store.prefix(ids)
    k = pre.point(n)
    if store.has_point(k):
        return
    if not force and store.full():
        if all(u[0] != k for u in dec.unsaved):
            dec.unsaved.append((k, list(ids), st, snap, tail))
            store.defer(k, n)
        return
    main, mtp = _caches(st)
    cp = store.copier
    m = int(snap["mtp_len"]) if mtp else n
    if mtp and st.mtp_len < m:
        return
    full = min((n - 1) // B, m // B if mtp else n)
    order = [pre.block(j) for j in range(full)]
    blocks = {h: {"main": _take(main, j * B, (j + 1) * B, st.ratio, cp), "mtp": _take(mtp, j * B, (j + 1) * B, st.ratio, cp)}
              for j, h in enumerate(order) if not store.has_block(h)}
    point = {"ids": torch.tensor(ids, dtype=torch.int32), "blocks": order, "pos": n, "mtp_len": m,
             "main": _take(main, full * B, n, st.ratio, cp), "mtp": _take(mtp, full * B, m, st.ratio, cp),
             "rec": cp.host(snap["rec"]), "conv": cp.host(snap["conv"]), "ple_tail": cp.host(snap["ple_tail"]),
             "rope_delta": int(snap.get("rope_delta", 0) or 0)}
    if snap.get("ple_history") is not None:
        point["ple_history"] = torch.from_numpy(snap["ple_history"].copy())
    if tail is not None:
        point["tail"] = cp.host(tail)
    store.put(k, n, ids, point, blocks, order, _bytes(point) + sum(_bytes(b) for b in blocks.values()),
              ready=cp.done(), forced=force and store.full())


def _save(dec, u, force: bool) -> None:
    dec.unsaved = [x for x in dec.unsaved if x is not u]
    try:
        capture(dec, u[1], u[2], u[3], u[4], force=force)
    except Exception as exc:  # noqa: BLE001 - the disk copy never stops a request
        dec.store.stats["errors"] += 1
        print(f"[prompt-cache] save failed: {type(exc).__name__}: {exc}", flush=True)
    if all(x[0] != u[0] for x in dec.unsaved):
        dec.store.deferred.pop(u[0], None)


def retry(dec, *, force: bool = False) -> None:
    """Copy the points waiting in GPU slots, oldest first, while the queue has room (all of them: ``force``)."""

    for u in list(dec.unsaved):
        if not force and dec.store.full():
            return
        _save(dec, u, force)


def flush(dec, st, above: int = 0) -> None:
    """Slot ``st`` is about to be reset, reused or overwritten past ``above`` rows: copy its waiting points now."""

    for u in list(dec.unsaved):
        if u[2] is st and len(u[1]) > above:
            _save(dec, u, True)


def unsaved(dec, st, n: int) -> bool:
    return any(u[2] is st and len(u[1]) == n for u in dec.unsaved)


def restore(dec, prompt: list[int], best) -> None:
    """Load the longest stored point this prompt extends into a slot and keep it there, when that beats recomputing
    from ``best`` (the longest point in GPU memory). Keeps the busy slots and the memory one's chain."""

    store = dec.store
    pre = store.prefix(prompt)
    have = len(best[0]) if best is not None else 0
    hit = store.lookup(pre, have, len(prompt))
    if hit is None:
        return
    k, n, nbytes = hit
    busy = dec._busy()
    if dec.free:
        st = dec.free.pop()
        flush(dec, st)
    else:
        st = next((x[1] for x in dec.kept if id(x[1]) not in busy and (best is None or x[1] is not best[1])), None)
        if st is None:
            return
        dec._drop_kept(st)
    try:
        if not dec._grow(st, n + dec.depth + 2, protect=None if best is None else best[1]):
            dec.free.append(st)
            return
        point, blocks, t0 = store.read(k)
        if point["ids"].numel() != n or not torch.equal(point["ids"], torch.tensor(prompt[:n], dtype=torch.int32)):
            raise ValueError("a stored point's tokens differ from its key's")
        main, mtp = _caches(st)
        for j, blk in enumerate(blocks):
            _put(main, j * B, blk["main"], st.ratio)
            _put(mtp, j * B, blk["mtp"], st.ratio)
        full = len(blocks)
        _put(main, full * B, point["main"], st.ratio)
        _put(mtp, full * B, point["mtp"], st.ratio)
        dev = st.rec.device
        snap = {"pos": n, "rec": point["rec"].to(dev), "conv": point["conv"].to(dev),
                "ple_tail": point["ple_tail"].to(dev),
                "ple_history": point["ple_history"].numpy() if "ple_history" in point else None,
                "mtp_len": int(point["mtp_len"]), "image_positions": None, "rope_delta": point["rope_delta"],
                "vision_cache_key": None}
        st.restore(snap)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        store.loaded(k, n, nbytes, t0)
    except Exception as exc:  # noqa: BLE001 - a point that can't load is recomputed
        store.stats["errors"] += 1
        print(f"[prompt-cache] load failed, prefilling instead: {type(exc).__name__}: {exc}", flush=True)
        dec._shrink(st)
        dec.free.append(st)
        return
    dec._remember(list(prompt[:n]), st, snap, point["tail"].to(dev) if "tail" in point else None)
