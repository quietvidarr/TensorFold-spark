"""Select and retain prompt prefixes without consuming a longer chain when a spare slot can hold a copy."""

from tensorfold.cuda import keep_rank


def _ranks(owner) -> list[int]:
    """``keep_rank`` ranks of ``owner.kept`` (patch 0010): history points first, shared starting points last."""

    conts = getattr(owner, "conts", None) or {}
    waiting = getattr(owner, "unsaved", None) or ()
    return keep_rank.ranks([k[0] for k in owner.kept], [len(conts.get(keep_rank.point_key(k[0]), ())) for k in owner.kept],
                           [any(u[2] is k[1] and len(u[1]) == len(k[0]) for u in waiting) for k in owner.kept])


def slot_order(owner) -> list:
    """The kept slots, the one to free first first: its best point's rank lowest, then its oldest point oldest."""

    r = _ranks(owner)
    best: dict[int, tuple] = {}
    for i, k in enumerate(owner.kept):
        old = best.get(id(k[1]))
        best[id(k[1])] = (max(r[i], old[0]), min(i, old[1]), k[1]) if old else (r[i], i, k[1])
    return [v[2] for v in sorted(best.values(), key=lambda v: (v[0], v[1]))]


def _best(kept, prompt, busy=(), cache_key=None):
    return max((k for k in kept if id(k[1]) not in busy and len(k[0]) < len(prompt)
                and k[2].get("vision_cache_key") == cache_key
                and prompt[:len(k[0])] == k[0]), key=lambda k: len(k[0]), default=None)


def _longer(kept, entry):
    return any(k[1] is entry[1] and len(k[0]) > len(entry[0]) for k in kept)


def slot_for(owner, prompt: list[int], reuse: bool, cache_key=None):
    """Copy a fork into spare capacity; otherwise retain the released idle-slot and memory-pressure behavior."""

    busy = owner._busy()
    best = _best(owner.kept, prompt, cache_key=cache_key) if reuse else None
    fork = best is not None and (id(best[1]) in busy or _longer(owner.kept, best))
    if fork:
        if owner.free:
            spare = owner.free.pop()
            try:
                if owner._grow(spare, len(prompt) + owner.depth + 2, protect=best[1]):
                    spare.copy_prefix(best[1], len(best[0]), best[2]["mtp_len"])
                    return spare, {"state": best[2], "tail": best[3]}, len(best[0])
            except Exception:
                owner.free.append(spare)
                owner._shrink(spare, force=True)
                raise
            owner.free.append(spare)
        best = _best(owner.kept, prompt, busy, cache_key=cache_key) if reuse else None
        if best is not None and owner.free and _longer(owner.kept, best):
            best = None
    if best is not None:
        n = len(best[0])
        owner.kept = [k for k in owner.kept if k[1] is not best[1]
                      or len(k[0]) <= n and best[0][:len(k[0])] == k[0]]
        return best[1], {"state": best[2], "tail": best[3]}, n
    if not owner.free:
        idle = next((k[1] for k in owner.kept if id(k[1]) not in busy), None)
        if idle is None:
            raise RuntimeError("no free stream slot")
        owner._drop_kept(idle)
        owner.free.append(idle)
    return owner.free.pop(), None, 0


def remember(owner, ids, st, snap, tail) -> None:
    """Keep each slot's prefix chain, returning displaced idle slots to the free list."""

    key = snap.get("vision_cache_key")
    same = lambda k: k[0] == ids and k[2].get("vision_cache_key") == key
    gone = [k[1] for k in owner.kept if same(k)]
    owner.kept = [k for k in owner.kept if not same(k)] + [(ids, st, snap, tail)]
    while len(owner.kept) > owner.keep:
        # Over the cap (patch 0010, ``keep_rank``): a point inside a conversation's history goes first (it is on disk),
        # then a conversation's newest point, then a shared starting point (a reused system prompt, a fork's base);
        # the oldest first within each, and a point not yet copied for the disk last of all.
        r = _ranks(owner)
        at = min(range(len(owner.kept)), key=lambda i: (r[i], i))
        gone.append(owner.kept.pop(at)[1])
    busy = owner._busy()
    for old in gone:
        if old is not st and id(old) not in busy and all(k[1] is not old for k in owner.kept) and \
                all(f is not old for f in owner.free):
            owner.free.append(old)
