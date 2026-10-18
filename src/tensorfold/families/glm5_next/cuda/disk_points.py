"""GLM-5.3's kept prompt states to and from the prompt store on disk (``cuda/prompt_store.py``).

A point is a ``decode.Snapshot`` of ``GlmEngine.cache``: the KDA layers' recurrent and conv states at the point, the
pending MTP row, and the attention rows below it (``decode._row_views``: every DSA layer's latents, the indexer's keys,
gates and 4-token pools, the MTP head's rows below its own length). Rows go to the store in blocks of ``B`` tokens
(pools: ``B / 4`` rows), shared by the points of one conversation; the rows past the last whole block go with the
point. A loaded point joins the kept snapshots with its rows saved, and the request resumes from it as from any other.
"""

from __future__ import annotations

import torch

from tensorfold.cuda import prompt_store as ps

B = ps.B
POOL = 4                                            # tokens an indexer pool row covers (rows < n // 4 are complete)


def specs(st, n: int, m: int) -> list[tuple[torch.Tensor, int, int, bool]]:
    """``decode._row_views``'s tensors, in its order: (cache, rows the point holds, tokens a row, an MTP row)."""

    out = [(kc, n, 1, False) for kc in st.kc] + [(vc, n, 1, False) for vc in st.vc if vc is not None]
    idx = st.index or []
    for ik, ig, pk in idx[:len(st.kc)]:
        out += [(ik, n, 1, False), (ig, n, 1, False), (pk, n // POOL + 1, POOL, False)]
    if m > 0 and hasattr(st, "mtp_kc"):
        out.append((st.mtp_kc, m, 1, True))
        if getattr(st, "mtp_vc", None) is not None:
            out.append((st.mtp_vc, m, 1, True))
        if len(idx) > len(st.kc):
            ik, ig, pk = idx[-1]
            out += [(ik, m, 1, True), (ig, m, 1, True), (pk, m // POOL + 1, POOL, True)]
    return out


def _bytes(obj) -> int:
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_bytes(v) for v in obj)
    return 0


def geometry(st) -> str:
    """What the saved rows' layout depends on, beyond the model key: a different one never reads them."""

    shapes = [tuple(t.shape[1:]) for t, *_ in specs(st, 8, 7)]
    return (f"glm5_next|bf16|kda{tuple(st.rec.shape[1:])}|conv{tuple(st.conv.shape)}|rows{shapes}|"
            f"idx{len(st.index or [])}|B{B}")


def open_store(engine):
    return ps.open_store(geometry(engine.e.st), idle=lambda: not getattr(engine, "busy", False))


def _full(n: int, m: int) -> int:
    """Whole blocks a point of n tokens (m MTP rows) shares: every row of them is final and below both ends."""

    full = (n - 1) // B
    return min(full, m // B) if m > 0 else full


def capture(engine, snap, *, force: bool = False) -> bool:
    """Queue a kept snapshot for the store (its new blocks and the rows past them), unless it is there already. Its
    rows are read from the live caches (they must still hold its prefix) or from its saved rows. While the queue is
    full the snapshot waits in GPU memory (``engine.unsaved``) until ``retry`` finds room. True when queued or there."""

    store = engine.store
    ids = snap.ids
    n, m = len(ids), max(snap.mtp_len, 0)
    pre = store.prefix(ids)
    k = pre.point(n)
    if store.has_point(k):
        return True
    if snap.rows is None and engine.live[:n] != ids:
        return False                                   # its rows are gone (not ours to save any more)
    if not force and store.full():
        if all(u is not snap for u in engine.unsaved):
            engine.unsaved.append(snap)
            store.defer(k, n)
        return True
    cp = store.copier
    sp = specs(engine.e.st, n, m)
    src = snap.rows if snap.rows is not None else [t[:c] for t, c, _, _ in sp]
    if len(src) != len(sp):
        raise ValueError("a kept snapshot's rows do not match the caches' layout")
    full = _full(n, m)
    order = [pre.block(j) for j in range(full)]
    blocks = {h: {"t": [cp.host(v[j * B // q:(j + 1) * B // q]) for v, (_, _, q, _) in zip(src, sp)]}
              for j, h in enumerate(order) if not store.has_block(h)}
    point = {"ids": torch.tensor(ids, dtype=torch.int32), "blocks": order, "pos": n, "mtp_len": snap.mtp_len,
             "rows": [cp.host(v[full * B // q:c]) for v, (_, c, q, _) in zip(src, sp)],
             "rec": cp.host(snap.rec, private=True), "conv": cp.host(snap.conv, private=True),
             "pending": cp.host(snap.pending, private=True) if snap.pending is not None else None}
    store.put(k, n, ids, point, blocks, order, _bytes(point) + sum(_bytes(b) for b in blocks.values()),
              ready=cp.done(), forced=force and store.full())
    return True


def save(engine, snap, *, force: bool = False) -> None:
    try:
        capture(engine, snap, force=force)
    except Exception as exc:  # noqa: BLE001 - the disk copy never stops a request
        engine.store.stats["errors"] += 1
        print(f"[prompt-cache] save failed: {type(exc).__name__}: {exc}", flush=True)


def retry(engine, *, force: bool = False, only=None) -> None:
    """Copy the waiting snapshots, oldest first, while the queue has room (all of them: ``force``; ``only``: those
    whose rows the next prefill overwrites, copied now whatever the queue holds)."""

    for snap in list(engine.unsaved):
        urgent = only is not None and snap in only
        if not force and not urgent and engine.store.full():
            continue
        engine.unsaved = [x for x in engine.unsaved if x is not snap]
        save(engine, snap, force=force or urgent)
        k = engine.store.prefix(snap.ids).point(len(snap.ids))
        if all(x is not snap for x in engine.unsaved):
            engine.store.deferred.pop(k, None)


def restore(engine, prompt: list[int], have: int):
    """The longest stored point this prompt extends, as a ``decode.Snapshot`` with its rows saved (on the GPU), when
    that beats recomputing from the ``have`` tokens a kept snapshot already covers; else None."""

    from .decode import Snapshot

    store = engine.store
    pre = store.prefix(prompt)
    hit = store.lookup(pre, have, len(prompt))
    if hit is None:
        return None
    k, n, nbytes = hit
    try:
        point, blocks, t0 = store.read(k)
        if point["ids"].numel() != n or not torch.equal(point["ids"], torch.tensor(prompt[:n], dtype=torch.int32)):
            raise ValueError("a stored point's tokens differ from its key's")
        st = engine.e.st
        m = max(int(point["mtp_len"]), 0)
        sp = specs(st, n, m)
        if len(point["rows"]) != len(sp):
            raise ValueError("a stored point's rows do not match the caches' layout")
        dev = st.conv.device
        rows = []
        for i, (t, c, q, _) in enumerate(sp):
            r = torch.cat([b["t"][i] for b in blocks] + [point["rows"][i]]).to(dev)
            if r.shape[0] != c or r.shape[1:] != t.shape[1:] or r.dtype != t.dtype:
                raise ValueError(f"a stored point's rows {tuple(r.shape)} do not fit {c} rows of {tuple(t.shape)}")
            rows.append(r)
        rec, conv = point["rec"].to(dev), point["conv"].to(dev)
        if rec.shape != st.rec.shape[1:] or conv.shape != st.conv.shape:
            raise ValueError("a stored point's recurrent states do not fit the engine's")
        pending = point["pending"].to(dev) if point["pending"] is not None else None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        store.loaded(k, n, nbytes, t0)
    except Exception as exc:  # noqa: BLE001 - a point that can't load is recomputed
        store.stats["errors"] += 1
        print(f"[prompt-cache] load failed, prefilling instead: {type(exc).__name__}: {exc}", flush=True)
        return None
    snap = Snapshot(list(prompt[:n]), rec, conv, pending, int(point["mtp_len"]), -1)
    snap.rows = rows
    snap.nbytes = sum(r.numel() * r.element_size() for r in rows)
    return snap
