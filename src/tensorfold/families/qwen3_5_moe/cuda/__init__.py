"""Qwen3.6 MoE on CUDA: MTP drafting settings."""

import os

DEPTH = 3            # most MTP drafts a round
CONFIDENCE = 0.3     # a chain ends after a draft the MTP head gives less than this
COALESCE = os.environ.get("TF_COALESCE", "1") != "0"     # one read-back a draft level (0: three to five)


def _by_n(text: str) -> dict:
    """TF_MTP_BY_N="1:8:0.6;2:3:0.3[:before]": (depth, confidence, drop-the-low-draft) by streams in the round; the
    entry with the largest count <= the round's applies, its depth capped at the engine's (--mtp-drafts)."""

    out = {}
    for part in filter(None, (x.strip() for x in text.split(";"))):
        n, d, c, *mode = part.split(":")
        out[int(n)] = (int(d), float(c), (mode[0] if mode else "after") == "before")
    return out


BY_N = _by_n(os.environ.get("TF_MTP_BY_N", ""))


def policy_for(n: int, depth: int, confidence: float) -> tuple[int, float, bool]:
    """(depth, confidence, before) for a round of ``n`` streams: TF_MTP_BY_N's entry, else the engine's settings."""

    keys = [k for k in BY_N if k <= n]
    if not keys:
        return depth, confidence, False
    d, c, b = BY_N[max(keys)]
    return min(d, depth), c, b
