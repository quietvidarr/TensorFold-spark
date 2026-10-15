"""Which kept prompt point leaves GPU memory first (local patch 0010). Family-independent: a decoder passes its kept
points' token ids (oldest first), how many different continuations requests resumed each one with, and which ones are
not yet copied for the disk store.

Ranks, lowest dropped first, the oldest first within a rank:

  0  a point inside one conversation's history: a newer kept point of the same conversation extends it. With a point
     at every message start these are most of them, and each is in the disk store already.
  1  a conversation's newest point (where its next turn resumes) or a one-off prompt's end.
  2  a shared starting point: a first-message point (a system prompt plus tools) that a request resumed from, a point
     that requests resumed from with 2 or more different continuations (forks, regenerates, edits, other
     conversations), or the common prefix of 2 or more kept points that went different ways.
  3  a point not yet copied for the disk store (its only copy is this GPU state): dropped only when nothing else is
     left, and then copied first.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

OPENERS = np.zeros(0, dtype=np.int64)        # message-start token ids (set by ``markers.snapshot_points``)
CONT = 16                                    # tokens past a point that tell its continuations apart


def extends(a: Sequence[int], b: Sequence[int]) -> bool:
    """``a`` strictly extends ``b``."""

    n = len(b)
    if len(a) <= n or (n and a[n - 1] != b[n - 1]):
        return False
    return a[:n] == b if isinstance(a, list) and isinstance(b, list) else list(a[:n]) == list(b)


def first_point(ids: Sequence[int]) -> bool:
    """A point at the end of a prompt's first message (one message opener in it: a system prompt plus tools)."""

    if not len(OPENERS) or not len(ids):
        return False
    return int(np.isin(np.asarray(ids, dtype=np.int64), OPENERS).sum()) == 1


def ranks(ids: list[Sequence[int]], conts: list[int], unsaved: list[bool]) -> list[int]:
    out = []
    for i, a in enumerate(ids):
        if unsaved[i]:
            out.append(3)
            continue
        ext = [j for j, o in enumerate(ids) if j != i and extends(o, a)]
        leaves = [j for j in ext if not any(m != j and extends(ids[m], ids[j]) for m in ext)]
        shared = conts[i] >= 2 or len(leaves) >= 2 or (conts[i] >= 1 and first_point(a))
        out.append(2 if shared else 0 if ext else 1)
    return out


def victim(ids: list[Sequence[int]], conts: list[int], unsaved: list[bool], among: Sequence[int] | None = None) -> int:
    """The index (into ``ids``, oldest first) of the point to drop: the lowest rank, then the oldest."""

    r = ranks(ids, conts, unsaved)
    cand = range(len(ids)) if among is None else among
    return min(cand, key=lambda i: (r[i], i))


def point_key(ids: Sequence[int], n: int | None = None) -> tuple:
    """A kept point's identity for ``conts`` (its length and last tokens)."""

    n = len(ids) if n is None else n
    return (n, tuple(int(x) for x in ids[max(0, n - CONT):n]))


def note(conts: dict, prompt: Sequence[int], n: int, keep: list | None = None) -> None:
    """A request resumed from the point at ``n``: remember what it continued with (``keep``: the kept points' ids,
    to forget the points no longer kept)."""

    if n <= 0:
        return
    conts.setdefault(point_key(prompt, n), set()).add(cont_key(prompt, n))
    if keep is not None and len(conts) > 4 * len(keep) + 16:
        live = {point_key(ids) for ids in keep}
        for key in [k for k in conts if k not in live]:
            del conts[key]


def cont_key(prompt: Sequence[int], n: int) -> tuple:
    """What a request resuming from the point at ``n`` continues with (tells forks and conversations apart)."""

    return tuple(int(x) for x in prompt[n:n + CONT])
