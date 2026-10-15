"""Qwen3.6 MoE's concurrent decoder keeps every conversation's latest prompt state while any older one is left
(``KeptEnds``): several conversations at once keep resuming from their own last turn."""

from tensorfold.families.qwen3_5_moe.cuda.multi import KeptEnds


def _turns(cache, conversations: int, turns: int, system: int = 50) -> int:
    """Each turn of each conversation in turn: resume from its longest kept state, then keep its message start and
    its end (as a prefill does); returns how many turns found their previous end."""

    hits = 0
    prompts = {c: list(range(system)) for c in range(conversations)}
    for t in range(turns):
        for c in range(conversations):
            p = prompts[c] + [1000 * (c + 1) + 10 * t + j for j in range(20)]
            best = cache.longest(p)
            if t and best is not None and len(best[0]) > system:
                hits += 1
            cache.add(p[:len(prompts[c]) + 1], None, None)       # the new message's start
            cache.add(p[:-1], None, None)                       # the prompt end
            prompts[c] = p + [7, 7, 7]                          # the reply joins the next turn's prompt
    return hits


def test_each_conversation_keeps_its_latest_end():
    assert _turns(KeptEnds(8), conversations=8, turns=4) == 8 * 3          # every follow-up resumes
    assert _turns(KeptEnds(16), conversations=8, turns=4) == 8 * 3


def test_more_conversations_than_states_degrades_like_before():
    got = _turns(KeptEnds(4), conversations=8, turns=3)
    assert 0 <= got < 8 * 2


def test_shared_block_kept_until_extended_states_compete():
    cache = KeptEnds(3)
    cache.add(list(range(10)), None, None)                          # a system block
    cache.add(list(range(10)) + [1, 2], None, None)                 # conversation A
    cache.add(list(range(10)) + [3, 4], None, None)                 # conversation B
    cache.add(list(range(10)) + [5, 6], None, None)                 # C: over the cap, the extended block goes
    assert sorted(len(e[0]) for e in cache.entries) == [12, 12, 12]
