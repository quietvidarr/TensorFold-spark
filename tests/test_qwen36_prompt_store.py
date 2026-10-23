"""Local patch 0104: Qwen3.6 MoE's kept states to the prompt store on disk and back (a small State on the CPU)."""

import time
from types import SimpleNamespace

import pytest
import torch

from tensorfold.families.qwen3_5.cuda.forward import State
from tensorfold.families.qwen3_5.cuda.multi import kept
from tensorfold.families.qwen3_5_moe.cuda import disk_points
from tensorfold.families.qwen3_5_moe.cuda.mtp import Cache
from tensorfold.families.qwen3_5_moe.cuda.multi import KeptEnds


def _weights():
    cfg = SimpleNamespace(k_heads=2, dk=8, v_heads=4, dv=8, conv_kernel=4, kv_heads=2, head_dim=16, hidden=32)
    layers = [SimpleNamespace(linear=i % 4 != 3) for i in range(8)]
    return SimpleNamespace(config=cfg, layers=layers, norm=torch.zeros(1))


class _Dec:
    """The MultiDecoder surface disk_points uses."""

    def __init__(self, w):
        self.w, self.cache, self.streams, self.filling = w, KeptEnds(8), {}, []
        self.unsaved = []
        self.store = disk_points.open_store(self)

    def add(self, ids, state, snap):
        disk_points.capture(self, ids, state, snap)
        self.cache.add(ids, state, snap)


def _state(w, n, seed):
    g = torch.Generator().manual_seed(seed)
    st = State(w)
    c = w.config
    st.kv = [None if kv is None else (torch.rand((n + 50, c.kv_heads, c.head_dim), generator=g).bfloat16(),
                                      torch.rand((n + 50, c.kv_heads, c.head_dim), generator=g).bfloat16())
             for kv in st.kv]
    st.rec = [None if r is None else torch.rand(r.shape, generator=g) for r in st.rec]
    st.conv = [None if x is None else torch.rand(x.shape, generator=g).bfloat16() for x in st.conv]
    st.pos = n
    mc = Cache(w, n + 50)
    mc.k.copy_(torch.rand(mc.k.shape, generator=g))
    mc.v.copy_(torch.rand(mc.v.shape, generator=g))
    mc.pos = n - 1
    return kept(st), (mc.view(), torch.rand((2, c.hidden), generator=g).bfloat16())


def _drain(store):
    for _ in range(200):
        if store.q.empty() and not store.pending:
            return
        time.sleep(0.05)
    raise AssertionError("writer did not finish")


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("TF_PROMPT_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TF_PROMPT_CACHE_KEY", "test-model")
    monkeypatch.setenv("SERVED_NAME", "test")
    monkeypatch.setenv("TF_PROMPT_CACHE_READ_GBS", "100")
    return tmp_path


def test_round_trip(env):
    w = _weights()
    a = _Dec(w)
    n = 2600
    ids = list(range(1000, 1000 + n))
    st, (mc, held) = _state(w, n, 1)
    a.add(ids, st, (mc, held))
    _drain(a.store)
    assert len(a.store.points) == 1 and len(a.store.blocks) == 2        # two whole blocks, 552 rows with the point

    b = _Dec(w)                                                         # a new server: the index comes from disk
    disk_points.restore(b, ids + [5, 6, 7])
    assert len(b.cache.entries) == 1
    got_ids, dst, (got_mc, got_held) = b.cache.entries[0]
    assert got_ids == ids and dst.pos == n and got_mc.pos == n - 1
    for x, y in zip(st.kv, dst.kv):
        if x is None:
            assert y is None
            continue
        assert torch.equal(x[0][:n], y[0]) and torch.equal(x[1][:n], y[1])
    for x, y in zip(st.rec + st.conv, dst.rec + dst.conv):
        assert (x is None and y is None) or torch.equal(x, y)
    assert torch.equal(mc.k[:n - 1], got_mc.k[:n - 1]) and torch.equal(mc.v[:n - 1], got_mc.v[:n - 1])
    assert torch.equal(held, got_held)
    assert b.cache.longest(ids + [5])[0] == ids                         # a request resumes from it


def test_shared_blocks_and_gpu_first(env):
    w = _weights()
    a = _Dec(w)
    ids = list(range(5000))
    st, snap = _state(w, 4000, 2)
    a.add(ids[:2100], kept(_cut(st, 2100)), (_cut_mc(snap[0], 2099), snap[1]))
    a.add(ids[:4000], st, snap)
    _drain(a.store)
    assert len(a.store.points) == 2 and len(a.store.blocks) == 3        # blocks 0-1 shared, block 2 the longer one's
    b = _Dec(w)
    b.cache.add(ids[:4000], st, snap)                                   # the GPU already holds it: no load
    disk_points.restore(b, ids)
    assert len(b.cache.entries) == 1 and b.store.stats["loads"] == 0
    c = _Dec(w)
    disk_points.restore(c, ids[:3000])                                  # the longest stored point below the prompt
    assert [len(e[0]) for e in c.cache.entries] == [2100]


def _cut(st, n):
    other = kept(st)
    other.pos = n
    return other


def _cut_mc(mc, pos):
    other = mc.view()
    other.pos = pos
    return other


def test_full_queue_waits_in_gpu_never_drops(env, monkeypatch):
    """Patch 0107: past the queue's cap a kept state waits in GPU memory and is copied when there is room."""
    monkeypatch.setenv("TF_PROMPT_CACHE_QUEUE_GIB", "1e-9")
    w = _weights()
    a = _Dec(w)
    store = a.store
    store.queued = store.queue_cap                                          # the writer is behind
    for j in range(3):
        state, snap = _state(w, 1100 + 300 * j, j)
        a.add(list(range(7 * j, 7 * j + 1100 + 300 * j)), state, snap)
    assert len(a.unsaved) == 3 and store.stats["deferred"] == 3 and store.stats["dropped"] == 0
    disk_points.retry(a)                                                    # still full
    assert len(a.unsaved) == 3
    store.queued -= store.queue_cap
    store.queue_cap = 1 << 40
    disk_points.retry(a)
    assert not a.unsaved
    _drain(store)
    assert len(store.points) == 3 and store.stats["lost"] == 0
    full = _Dec(w)                                                          # forced copies go past the cap
    full.store.queued = full.store.queue_cap = 1
    state, snap = _state(w, 1300, 9)
    disk_points.capture(full, list(range(500, 1800)), state, snap, force=True)
    assert not full.unsaved and full.store.stats["forced"] == 1


def test_waiting_states_stay_within_their_cap(env, monkeypatch):
    """Patch 0011: with a slow writer the states waiting in GPU memory stay within TF_PROMPT_CACHE_DEFER_GIB; past
    it the step waits for the writer. Every state is saved."""
    monkeypatch.setenv("TF_PROMPT_CACHE_QUEUE_GIB", "1e-9")
    w = _weights()
    a = _Dec(w)
    s = a.store
    put = s.tier.put
    monkeypatch.setattr(s.tier, "put", lambda rel, obj: time.sleep(0.03) or put(rel, obj))
    states = [_state(w, 1100 + 300 * j, j) for j in range(6)]
    s.defer_cap = int(disk_points.held_bytes(*states[0]) * 1.5)        # room for one
    for j, (state, snap) in enumerate(states):
        a.add(list(range(7 * j, 7 * j + 1100 + 300 * j)), state, snap)
        assert sum(s.defer_bytes.values()) <= s.defer_cap and len(a.unsaved) <= 1 and len(s.inflight) <= 1
    assert s.stats["waits"] > 0
    disk_points.retry(a, force=True)
    _drain(s)
    assert not a.unsaved and not s.defer_bytes and len(s.points) == 6 and s.stats["lost"] == 0


def test_kept_ends_keep_shared_starts():
    """Over the cap: history states first, then conversations' newest; a reused system prompt's state stays."""
    import numpy as np
    from tensorfold.cuda import keep_rank
    keep_rank.OPENERS = np.array([1])
    c = KeptEnds(3)
    sys_ = [1] + [7] * 300
    c.add(sys_, "s0", None)
    keep_rank.note(c.conts, sys_ + [1, 5], len(sys_))
    keep_rank.note(c.conts, sys_ + [1, 6], len(sys_))
    a1 = sys_ + [1, 8] + [3] * 40
    c.add(a1, "a1", None)
    c.add(a1 + [1, 9] + [3] * 40, "a2", None)                               # a1 is now history
    c.add(sys_ + [1, 11], "b", None)
    assert [e[1] for e in c.entries] == ["s0", "a2", "b"]
    c.waiting = lambda e: e[1] == "a2"
    c.add(sys_ + [1, 12], "c", None)
    assert [e[1] for e in c.entries] == ["s0", "a2", "c"]                   # "b" went; unsaved "a2" stayed
