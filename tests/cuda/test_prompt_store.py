"""Local patch 0006: kept prompt points to disk and back (a small State on the CPU), the store's keys, budgets and pins."""

import json
import sqlite3
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorfold.cuda import prompt_store as ps
from tensorfold.families.qwen4_exp.cuda import disk_points
from tensorfold.families.qwen4_exp.cuda.state import State


class _Ngram:
    def initial_history(self):
        return np.full((2,), 7, dtype=np.int64)


def _weights(mtp=True):
    cfg = SimpleNamespace(conv_kernel=4, conv_dim=48, nv=2, dv=8, dk=8, nk=2, index_ratio=4, index_dim=16, kv_heads=2,
                          head_dim=32, streams=2, hidden=16, ple_kernel=3, ngram_size=2, ple_layers=1,
                          ngram=lambda i: _Ngram())
    layers = [SimpleNamespace(linear=i % 4 != 3, index=i) for i in range(8)]
    return SimpleNamespace(cfg=cfg, device="cpu", layers=layers, mtp=object() if mtp else None)


def _fill(st, n, seed):
    g = torch.Generator().manual_seed(seed)
    for c in [*st.kc, *([st.mtp_kc] if hasattr(st, "mtp_kc") else [])]:
        for name in ("k", "v", "ks", "vs"):
            t = getattr(c, name)
            if t.shape[0] > 1:
                t[:n].copy_((torch.rand(t[:n].shape, generator=g) * 100).to(t.dtype))
    for t in [*st.ikc, *st.pooled, *([st.mtp_ikc, st.mtp_pooled] if hasattr(st, "mtp_kc") else [])]:
        t.copy_(torch.rand(t.shape, generator=g).to(t.dtype))
    st.rec.copy_(torch.rand(st.rec.shape, generator=g))
    st.conv.copy_(torch.rand(st.conv.shape, generator=g).to(st.conv.dtype))
    st.ple_tail.copy_(torch.rand(st.ple_tail.shape, generator=g).to(st.ple_tail.dtype))
    st.set_pos(n)
    st.set_mtp_len(n - 1)


class _Dec:
    """The MultiDecoder surface disk_points uses."""

    def __init__(self, w, kv, slots=2, cap=8192):
        self.free = [State(w, cap, 4, kv) for _ in range(slots)]
        self.kept, self.streams, self.filling, self.depth = [], {}, [], 3
        self.unsaved, self.conts = [], {}
        self.store = disk_points.open_store(self)

    def _busy(self):
        return set()

    def _grow(self, st, rows, alone=False, protect=None):
        return rows <= st.capacity

    def _drop_kept(self, st):
        self.kept = [k for k in self.kept if k[1] is not st]

    def _shrink(self, st, **kw):
        pass

    def _remember(self, ids, st, snap, tail):
        disk_points.capture(self, ids, st, snap, tail)
        self.kept.append((ids, st, snap, tail))


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


@pytest.mark.parametrize("kv", ["bf16", "int8"])
def test_point_round_trip(env, kv):
    w = _weights()
    a = _Dec(w, kv)
    src = a.free[0]
    n = 2600
    _fill(src, n, 1)
    ids = list(range(1000, 1000 + n))
    snap = {**src.snapshot(), "mtp_len": n - 1}
    tail = torch.rand((1, 32))
    a._remember(ids, src, snap, tail)
    _drain(a.store)
    assert len(a.store.points) == 1 and len(a.store.blocks) == 2          # two whole blocks, 552 rows with the point

    b = _Dec(w, kv)                                                       # a new server: the index comes from disk
    assert len(b.store.points) == 1
    prompt = ids + [5, 6, 7]
    disk_points.restore(b, prompt, None)
    assert len(b.kept) == 1
    got_ids, dst, got_snap, got_tail = b.kept[0]
    assert got_ids == ids and dst.pos == n and dst.mtp_len == n - 1
    for x, y in zip([*src.kc, src.mtp_kc], [*dst.kc, dst.mtp_kc]):
        rows = n if x is not src.mtp_kc else n - 1
        assert torch.equal(x.k[:rows], y.k[:rows]) and torch.equal(x.v[:rows], y.v[:rows])
        if x.quantized:
            assert torch.equal(x.ks[:rows], y.ks[:rows])
    for x, y in zip(src.ikc, dst.ikc):
        assert torch.equal(x[:n], y[:n])
    for x, y in zip(src.pooled, dst.pooled):
        assert torch.equal(x[:n // 4], y[:n // 4])
    assert torch.equal(src.mtp_pooled[:(n - 1) // 4], dst.mtp_pooled[:(n - 1) // 4])
    assert torch.equal(dst.rec[0], snap["rec"]) and torch.equal(dst.conv, snap["conv"])
    assert torch.equal(got_tail, tail)
    assert np.array_equal(got_snap["ple_history"], snap["ple_history"])


def test_shared_blocks_and_lookup(env):
    w = _weights()
    a = _Dec(w, "bf16")
    st = a.free[0]
    _fill(st, 5000, 2)
    ids = list(range(5000))
    for n in (1500, 2600, 4100):                      # one conversation's points: its blocks are written once
        a._remember(ids[:n], st, {**st.snapshot(), "pos": n, "mtp_len": n - 1}, None)
    _drain(a.store)
    assert len(a.store.points) == 3 and len(a.store.blocks) == 4
    pre = a.store.prefix(ids[:4500])
    assert a.store.lookup(pre, 0, 4500)[1] == 4100
    a.store.read_bps = 1e8                                           # a slow disk (100 MB/s):
    assert a.store.lookup(pre, 4000, 4500) is None                   # 100 tokens: recomputing is quicker
    other = a.store.prefix(ids[:2000] + [9] * 2500)
    assert a.store.lookup(other, 0, 4500)[1] == 1500                 # diverges at 2000: the 1500 point
    assert a.store.lookup(a.store.prefix(ids[:2600]), 0, 2600)[1] == 1500    # a point must leave a token to prefill


def test_pin_ttl_and_budget(env, monkeypatch):
    monkeypatch.setenv("TF_PROMPT_CACHE_PIN_AFTER", "2")
    w = _weights()
    a = _Dec(w, "bf16")
    st = a.free[0]
    _fill(st, 3000, 3)
    ids = list(range(3000))
    a._remember(ids[:1200], st, {**st.snapshot(), "pos": 1200, "mtp_len": 1199}, None)
    a._remember(ids[:2900], st, {**st.snapshot(), "pos": 2900, "mtp_len": 2899}, None)
    _drain(a.store)
    s = a.store
    s.used(s.prefix(ids[:1200] + [1] * 100), 1200)
    s.used(s.prefix(ids[:1200] + [1] * 100), 1200)                 # the same continuation again
    s.used(s.prefix(ids[:1200] + [2] * 100), 1200)                 # a second one: pinned
    time.sleep(0.5)
    db = sqlite3.connect(env / "index.sqlite")
    rows = dict(db.execute("SELECT n, pinned FROM points"))
    assert rows == {1200: 1, 2900: 0}
    assert json.loads(db.execute("SELECT conts FROM points WHERE n=1200").fetchone()[0]).__len__() == 2
    s.ttl = 0                                                       # every normal point is past its TTL
    s.q.put(("used", "none", "x", None))
    time.sleep(0.5)
    _drain(s)
    s._evict(s._db())
    assert dict(sqlite3.connect(env / "index.sqlite").execute("SELECT n, pinned FROM points")) == {1200: 1}
    assert len(s.points) == 1 and len(s.blocks) == 1                 # the 2900 point's own block went with it
    assert len(list((env / s.key / "b").glob("*.pt"))) == 1
    s.pinned_cap = 1                                                # over the pinned budget: the pin goes too
    s._evict(s._db())
    assert not s.points and not list((env / s.key / "p").glob("*.pt"))


def test_new_key_rebuilds_pins(env, monkeypatch):
    monkeypatch.setenv("TF_PROMPT_CACHE_PIN_AFTER", "1")
    w = _weights()
    a = _Dec(w, "bf16")
    st = a.free[0]
    _fill(st, 1500, 4)
    ids = list(range(1500))
    a._remember(ids[:1300], st, {**st.snapshot(), "pos": 1300, "mtp_len": 1299}, None)
    _drain(a.store)
    a.store.used(a.store.prefix(ids[:1300] + [3] * 10), 1300)
    time.sleep(0.5)
    monkeypatch.setenv("TF_PROMPT_CACHE_KEY", "test-model-v2")
    b = _Dec(w, "bf16")                                           # (no idle callback here: nothing replays)
    assert len(b.store.repin) == 1 and not b.store.points
    _fill(b.free[1], 1500, 5)                                     # the replay's prefill saves the same tokens again
    b._remember(ids[:1300], b.free[1], {**b.free[1].snapshot(), "pos": 1300, "mtp_len": 1299}, None)
    _drain(b.store)
    db = sqlite3.connect(env / "index.sqlite")
    assert db.execute("SELECT model, pinned FROM points").fetchall() == [(b.store.key, 1)]
    assert not (env / a.store.key / "p").exists() or not list((env / a.store.key / "p").glob("*.pt"))


def test_points_at_last_message_starts(monkeypatch):
    from tensorfold.cuda import markers
    monkeypatch.setattr(markers, "LAST", 2)
    opener, asst = 1, 2
    points = markers.snapshot_points([opener], [opener, asst])
    ids = [opener, 9] + [5] * 400 + [opener, 9] + [5] * 400 + [opener, asst] + [5] * 400 + [opener, 9] + \
          [5] * 300 + [opener, asst]
    starts = [i for i, t in enumerate(ids) if t == opener]
    got = points(ids)
    assert starts[1] in got and starts[3] in got and starts[4] in got      # system, the last user message, the reply
    monkeypatch.setattr(markers, "LAST", 0)
    assert starts[3] not in points(ids)


def test_points_at_every_message_start(monkeypatch):
    """Patch 0010: TF_POINTS_LAST=all keeps a point at every message start, and every start has a point at most
    MIN_GAP tokens before it (two starts closer than that: the earlier one stays)."""
    from tensorfold.cuda import markers
    monkeypatch.setattr(markers, "LAST", -1)
    opener, asst = 1, 2
    points = markers.snapshot_points([opener], [opener, asst])
    sizes = [700, 300, 20, 900, 15, 30, 500, 10, 400, 260]                 # tiny messages among long ones
    ids = [opener, 9] + [5] * 600
    for i, n in enumerate(sizes):
        ids += [opener, asst if i % 2 else 9] + [5] * n
    ids += [opener, asst]
    starts = [i for i, t in enumerate(ids) if t == opener][1:]
    got = points(ids)
    assert all(b - a >= markers.MIN_GAP for a, b in zip(got, got[1:]))
    for s in starts:
        assert any(s - markers.MIN_GAP < p <= s for p in got), (s, got)
    long_ones = [s for s, n in zip(starts, [600] + sizes) if n >= markers.MIN_GAP]
    assert set(long_ones) <= set(got)                                       # after a long message: exactly there
    monkeypatch.setattr(markers, "LAST", 3)
    assert len(points(ids)) < len(got)
    monkeypatch.setattr(markers, "LAST", -1)
    monkeypatch.setattr(markers, "GAP", 1024)                               # older starts thinned, last 3 kept
    thin = points(ids)
    assert len(thin) < len(got) and set(got[-3:]) <= set(thin)
    assert all(b - a >= 1024 for a, b in zip(thin[:-3], thin[1:-3]))


def test_full_queue_defers_never_drops(env, monkeypatch):
    """Patch 0010: past the queue's cap a point waits in its GPU slot, is copied when the queue has room (retry) or at
    once when its slot is needed (flush), and is never dropped."""
    monkeypatch.setenv("TF_PROMPT_CACHE_QUEUE_GIB", "1e-9")
    w = _weights()
    a = _Dec(w, "bf16", slots=3)
    store = a.store
    sts = [a.free.pop() for _ in range(3)]
    snaps = []
    for j, st in enumerate(sts):
        _fill(st, 1300 + j, j)
        ids = list(range(10 * j, 10 * j + 1300 + j))
        snaps.append((ids, {**st.snapshot(), "mtp_len": 1299 + j}))
    store.queued = store.queue_cap                                          # the writer is behind
    for st, (ids, snap) in zip(sts, snaps):
        disk_points.capture(a, ids, st, snap, None)
    assert len(a.unsaved) == 3 and store.stats["deferred"] == 3 and store.stats["dropped"] == 0
    disk_points.flush(a, sts[1])                                           # slot 1 is about to be reused
    assert len(a.unsaved) == 2 and store.stats["forced"] == 1
    disk_points.retry(a)                                                    # still full: nothing moves
    assert len(a.unsaved) == 2
    _drain(store)
    store.queued -= store.queue_cap
    store.queue_cap = 1 << 40                                               # room again
    disk_points.retry(a)
    assert not a.unsaved
    _drain(store)
    assert len(store.points) == 3 and store.stats["lost"] == 0 and not store.deferred


def test_journal_counts_points_lost_at_a_kill(env):
    w = _weights()
    a = _Dec(w, "bf16")
    st = a.free[0]
    _fill(st, 1200, 3)
    ids = list(range(1200))
    a._remember(ids, st, {**st.snapshot(), "mtp_len": 1199}, None)
    _drain(a.store)
    a.store._journal("q deadbeef 4096")                                     # queued, then the process was killed
    a.store.defer("cafe", 512)                                              # waited in a GPU slot, never copied
    b = _Dec(w, "bf16")
    assert b.store.stats["lost_before"] == 2 and len(b.store.points) == 1
    c = _Dec(w, "bf16")                                                     # counted once
    assert c.store.stats["lost_before"] == 0


def test_close_waits_for_the_writer(env):
    w = _weights()
    a = _Dec(w, "bf16")
    st = a.free[0]
    _fill(st, 3000, 4)
    a._remember(list(range(3000)), st, {**st.snapshot(), "mtp_len": 2999}, None)
    a.store.close()
    assert len(a.store.points) == 1 and a.store.stats["lost"] == 0
    assert json.loads((env / a.store.key / "stats.json").read_text())["saved"] == 1


def test_keep_rank():
    from tensorfold.cuda import keep_rank
    keep_rank.OPENERS = np.array([1])
    sys_ = [1, 7, 7]                                                        # a system prompt's point
    a1, a2 = sys_ + [1, 8], sys_ + [1, 8, 8, 1, 8]                          # conversation A: history, newest
    b1 = sys_ + [1, 9]                                                      # conversation B: newest
    ids = [sys_, a1, a2, b1]
    r = keep_rank.ranks(ids, [0, 0, 0, 0], [False] * 4)
    assert r == [2, 0, 1, 1]                                                # sys_: base of 2 diverging chains
    r = keep_rank.ranks([sys_, a1, a2], [0, 0, 0], [False] * 3)
    assert r == [0, 0, 1]                                                   # one conversation: history points
    r = keep_rank.ranks([sys_, a1, a2], [1, 0, 0], [False] * 3)
    assert r[0] == 2                                                        # a first-message point resumed from
    r = keep_rank.ranks([sys_, a1, a2], [0, 2, 0], [False, False, True])
    assert r == [0, 2, 3]                                                   # 2 continuations; unsaved last
    assert keep_rank.victim(ids, [0, 0, 0, 0], [False] * 4) == 1
    conts = {}
    keep_rank.note(conts, a2 + [5], len(a1))
    keep_rank.note(conts, a1 + [6, 6], len(a1))
    assert len(conts[keep_rank.point_key(a1)]) == 2


def test_gpu_points_keep_shared_starts(monkeypatch):
    """Flash's kept points over the cap: history points go first, then one-off ends; a shared start stays."""
    from tensorfold.cuda import keep_rank
    from tensorfold.families.qwen4_exp.cuda import prefixes
    keep_rank.OPENERS = np.array([1])
    slots = [object() for _ in range(6)]
    owner = SimpleNamespace(kept=[], keep=3, free=[], conts={}, unsaved=[], _busy=lambda: set())
    sys_ = [1] + [7] * 300
    prefixes.remember(owner, sys_, slots[0], {}, None)
    keep_rank.note(owner.conts, sys_ + [1, 5], len(sys_))                   # two conversations started from it
    keep_rank.note(owner.conts, sys_ + [1, 6], len(sys_))
    for i in range(1, 6):                                                   # five one-off conversations' ends
        prefixes.remember(owner, sys_ + [1, 10 + i] + [3] * 50, slots[i], {}, None)
    kept = [k[0] for k in owner.kept]
    assert sys_ in kept and len(kept) == 3
    assert [k[1] for k in owner.kept][1:] == slots[4:]                      # the newest ends stay


def _old_pool(db, pinned):
    """The pools as patch 0006 summed them (a correlated EXISTS per block): what ``_pools`` must equal."""
    pts = db.execute("SELECT COALESCE(SUM(bytes),0) FROM points WHERE pinned=?", (pinned,)).fetchone()[0]
    held = ("EXISTS" if pinned else "NOT EXISTS") + (
        " (SELECT 1 FROM uses u JOIN points p ON p.model=u.model AND p.key=u.point "
        "WHERE u.model=b.model AND u.block=b.hash AND p.pinned=1)")
    return pts + db.execute(f"SELECT COALESCE(SUM(bytes),0) FROM blocks b WHERE {held}").fetchone()[0]


def test_pools_equal_the_old_sums_and_are_quick(env):
    """Patch 0011: ``_pools`` by one join equals the old correlated sums (two models in one index, blocks shared by
    pinned and normal points, uses of a block one model has not), on an index the size of a full one."""
    w = _weights()
    s = _Dec(w, "bf16").store
    db = s._db()
    rng = np.random.default_rng(0)
    db.execute("BEGIN")
    for model in ("m1", "m2"):
        blocks = [f"b{j}" for j in range(300)]
        db.executemany("INSERT INTO blocks VALUES (?,?,?)", [(model, h, int(rng.integers(1, 1 << 30))) for h in blocks])
        for i in range(350):
            db.execute("INSERT INTO points(model,key,n,bytes,total,created,used,pinned) VALUES (?,?,?,?,?,?,?,?)",
                       (model, f"p{i}", i, int(rng.integers(1, 1 << 26)), 0, 0, i, int(i % 40 == 0)))
            start = int(rng.integers(0, 250))
            uses = [(model, f"p{i}", f"b{j}") for j in range(start, start + int(rng.integers(1, 50)))]
            db.executemany("INSERT OR IGNORE INTO uses VALUES (?,?,?)", uses + [(model, f"p{i}", "gone")])
    db.execute("COMMIT")
    t0 = time.perf_counter()
    old = {p: _old_pool(db, p) for p in (0, 1)}
    t1 = time.perf_counter()
    new = s._pools(db)
    t2 = time.perf_counter()
    assert new == old and old[1] > 0
    assert t2 - t1 < max(0.2, (t1 - t0) / 2)


def test_puts_scan_the_pools_only_near_a_cap(env, monkeypatch):
    """Patch 0011: a put runs the eviction scan once the bytes written since the last one may pass a cap, not after
    every put; the caps still hold."""
    w = _weights()
    a = _Dec(w, "bf16")
    s = a.store
    calls = []
    real = s._pools
    monkeypatch.setattr(s, "_pools", lambda db: calls.append(1) or real(db))
    st = a.free[0]
    _fill(st, 6000, 7)
    ids = list(range(6000))
    for n in (1100, 2200, 3300, 4400):
        a._remember(ids[:n], st, {**st.snapshot(), "pos": n, "mtp_len": n - 1}, None)
    _drain(s)
    time.sleep(0.2)
    assert len(s.points) == 4 and len(calls) == 1                  # the first put (nothing measured yet) only
    s.room = {0: 1, 1: 1}                                           # the pools all but full: the next put scans
    s.normal_cap = 1
    a._remember(ids[:5500], st, {**st.snapshot(), "pos": 5500, "mtp_len": 5499}, None)
    _drain(s)
    time.sleep(0.3)
    assert len(calls) >= 2 and not s.points                         # over the cap: evicted
