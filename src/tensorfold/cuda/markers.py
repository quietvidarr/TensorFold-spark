"""Message-start resume points for CUDA engines, found by the Mac planner's rule through the CUDA chat template."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

MIN_GAP = 256            # a snapshot this close to the prompt start or to the previous snapshot saves too little
# local patch 0006: also keep states at the last LAST message starts. A client that changes a recent message when it
# sends the next turn (e.g. one that adds recalled memories to the newest user message and leaves them out of the history)
# then resumes from that message's start instead of from the system prompt. 0: upstream's two points.
# local patch 0010: TF_POINTS_LAST=all keeps a state at EVERY message start (a fork, a regenerate or an edit at message
# k resumes at k's start). MIN_GAP still applies: of two starts closer than it the earlier one stays, so a resume
# point is never more than MIN_GAP tokens before any message start.
_last = os.environ.get("TF_POINTS_LAST", "0").strip().lower()
LAST = -1 if _last in ("all", "-1") else int(_last or 0)
# With ``all``, starts before a prompt's last 3 are kept only TF_POINTS_GAP tokens apart (default MIN_GAP: every one).
# Each kept point splits the prefill (about one more pass over the weights), so a long history arriving at once (an
# imported or edited chat) pays per point; a chat built turn by turn meets its new starts as its last ones anyway.
GAP = max(MIN_GAP, int(os.environ.get("TF_POINTS_GAP", "0") or 0))


class TemplateTokens:
    """The tokenizer surface ``message_markers`` reads, over a CUDA server's tokenizer and chat template."""

    def __init__(self, tok: Any, template: Any) -> None:
        self.tok, self.template = tok, template
        self.added_tokens_decoder = tok.get_added_tokens_decoder()
        self.all_special_ids = [int(i) for i, t in self.added_tokens_decoder.items() if t.special]

    def apply_chat_template(self, messages: list[dict[str, Any]], tokenize: bool = True,
                            add_generation_prompt: bool = False, **kwargs: Any):
        text = self.template.template.render(**self.template.specials, messages=messages,
                                             add_generation_prompt=add_generation_prompt, **kwargs)
        return self.tok.encode(text, add_special_tokens=False).ids if tokenize else text


def snapshot_points(openers: Sequence[int], assistant: Sequence[int]) -> Callable[[Sequence[int]], list[int]]:
    """Where a prefill keeps states: the second message's start (a shared system block) and the last assistant start."""

    from tensorfold.engine.prefill_plan import PrefillPlan

    plan = PrefillPlan(openers=openers, assistant=assistant, min_chunk=max(MIN_GAP, len(assistant)))
    from . import keep_rank
    keep_rank.OPENERS = np.asarray(sorted(set(int(o) for o in openers)), dtype=np.int64)

    def points(ids: Sequence[int]) -> list[int]:
        arr = np.asarray(ids, dtype=np.int64)
        found = plan.points(arr)
        starts = np.flatnonzero(np.isin(arr, plan.openers)) if plan.openers else []
        wanted = ([int(starts[1])] if len(starts) > 1 else []) + ([found[-1]] if found else [])
        if LAST < 0:
            last = -GAP
            for p in starts[1:-3]:
                if p - last >= GAP:
                    wanted.append(int(p))
                    last = int(p)
            wanted += [int(p) for p in starts[1:][-3:]]
        elif LAST > 0:
            wanted += [int(p) for p in starts[1:][-LAST:]]
        out: list[int] = []
        for p in sorted(set(wanted)):
            if MIN_GAP <= p < len(arr) and (not out or p - out[-1] >= MIN_GAP):
                out.append(p)
        return out

    return points


def resume_points(model_dir: str | Path) -> Callable[[Sequence[int]], list[int]] | None:
    """``snapshot_points`` for this checkpoint's chat template, or None when it marks no message starts."""

    from tokenizers import Tokenizer

    from tensorfold.engine.prefill_plan import message_markers

    from .chat_template import ChatTemplate

    model_dir = Path(model_dir)
    if not (model_dir / "tokenizer.json").is_file():
        return None
    try:
        tokens = TemplateTokens(Tokenizer.from_file(str(model_dir / "tokenizer.json")), ChatTemplate(model_dir))
        openers, assistant = message_markers(tokens)
    except (OSError, ValueError, KeyError):
        return None
    return snapshot_points(openers, assistant) if openers or assistant else None
