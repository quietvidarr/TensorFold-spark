"""GLM-5.3 prompt points to disk and back (disk_points.py, a small State on the CPU): rows, pools, MTP rows and
recurrent states come back bit for bit, shared blocks are written once, and a full queue keeps a point waiting."""

import time
from types import SimpleNamespace

import pytest
import torch

from tensorfold.cuda import prompt_store as ps
from tensorfold.families.glm5_next.cuda import disk_points
from tensorfold.families.glm5_next.cuda.decode import Snapshot, _row_views


def _state(cap=4096, dsa=3, latent=True, index=True, mtp=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: (torch.rand(s, generator=g) * 50).bfloat16()     # noqa: E731
    st = SimpleNamespace()
    st.kc = [r(cap, 16) if latent else r(cap, 2, 8) for _ in range(dsa)]
    st.vc = [None if latent else r(cap, 2, 8) for _ in range(dsa)]
    if mtp:
        st.mtp_kc = r(cap, 16)
        st.mtp_vc = None
    n_idx = dsa + (1 if mtp else 0)
    st.index = [(r(cap, 8), r(cap, 8), r(cap // 4 + 2, 8)) for _ in range(n_idx)] if index else None
    st.rec = torch.rand((2, 5, 2, 4, 4), generator=g)
    st.conv = torch.rand((5, 3, 24), generator=g).bfloat16()
    st.cur = [1] * 5
    return st


class _Eng:
    def __init__(self, st):
        self.e = SimpleNamespace(st=st)
        self.live, self.unsaved, self.later, self.busy = [], [], [], False
        self.store = disk_points.open_store(self)


def _snap(st, ids, mtp=True):
    n = len(ids)
    return Snapshot(list(ids), st.rec[st.cur[0]].clone(), st.conv.clone(),
                    torch.rand((1, 16)).bfloat16() if mtp else None, n - 1 if mtp else -1, -1)


def _wait(store, k, t=10.0):
    end = time.time() + t
    while k not in store.points and time.time() < end:
        time.sleep(0.02)
    assert k in store.points


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("TF_PROMPT_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TF_PROMPT_CACHE_KEY", "test-glm")
    monkeypatch.setenv("TF_PROMPT_CACHE_PREFILL_TPS", "1")      # any load beats recomputing in these tests
    return tmp_path


@pytest.mark.parametrize("n", [5, 1023, 1024, 1025, 2049, 3000])
@pytest.mark.parametrize("mtp", [True, False])
def test_point_round_trip_bit_for_bit(env, n, mtp):
    st = _state(mtp=mtp)
    eng = _Eng(st)
    ids = list(range(100, 100 + n))
    eng.live = ids + [7, 8]
    snap = _snap(st, ids, mtp)
    want = [v.clone() for v in _row_views(st, n, max(snap.mtp_len, 0))]
    assert disk_points.capture(eng, snap)
    k = eng.store.prefix(ids).point(n)
    _wait(eng.store, k)
    other = _Eng(_state(mtp=mtp, seed=1))                         # another engine (fresh caches) loads it
    got = disk_points.restore(other, ids + [1, 2, 3], 0)
    assert got is not None and got.ids == ids and got.mtp_len == snap.mtp_len
    assert len(got.rows) == len(want)
    for a, b in zip(got.rows, want):
        assert a.dtype == b.dtype and torch.equal(a, b)
    assert torch.equal(got.rec, snap.rec) and torch.equal(got.conv, snap.conv)
    assert (got.pending is None) == (snap.pending is None)
    if mtp:
        assert torch.equal(got.pending, snap.pending)


def test_points_share_blocks_and_saved_rows_work(env):
    st = _state()
    eng = _Eng(st)
    ids = list(range(5000, 5000 + 3500))
    eng.live = list(ids)
    a, b = _snap(st, ids[:2100]), _snap(st, ids[:3400])
    disk_points.capture(eng, a)
    disk_points.capture(eng, b)
    _wait(eng.store, eng.store.prefix(ids).point(3400))
    blocks = list((env / eng.store.key / "b").glob("*.pt"))
    assert len(blocks) == 3                       # blocks 0, 1 (shared) and 2 (b's): each written once
    # a snapshot whose rows were saved (another conversation took the caches) is copied from its saved rows
    c = _snap(st, ids[:1500])
    c.rows = [v.clone() for v in _row_views(st, 1500, 1499)]
    eng.live = [1, 2, 3]
    assert disk_points.capture(eng, c)
    _wait(eng.store, eng.store.prefix(ids).point(1500))
    got = disk_points.restore(_Eng(_state(seed=3)), ids[:1600], 0)
    assert got is not None and len(got.ids) == 1500
    assert all(torch.equal(x, y) for x, y in zip(got.rows, c.rows))
    # rows gone from the live caches and not saved: nothing to copy
    d = _snap(st, ids[:2900])
    assert not disk_points.capture(eng, d)


def test_full_queue_waits_then_copies(env, monkeypatch):
    st = _state()
    eng = _Eng(st)
    ids = list(range(9000, 9000 + 1200))
    eng.live = list(ids)
    monkeypatch.setattr(eng.store, "queue_cap", 0)              # full: the point waits on the GPU
    s = _snap(st, ids)
    assert disk_points.capture(eng, s)
    assert eng.unsaved == [s] and eng.store.deferred
    monkeypatch.setattr(eng.store, "queue_cap", 1 << 40)
    disk_points.retry(eng)
    assert eng.unsaved == [] and not eng.store.deferred
    _wait(eng.store, eng.store.prefix(ids).point(1200))


def test_layout_mismatch_is_refused(env):
    eng = _Eng(_state())
    ids = list(range(300, 1300))
    eng.live = list(ids)
    disk_points.capture(eng, _snap(eng.e.st, ids))
    _wait(eng.store, eng.store.prefix(ids).point(1000))
    other = _Eng(_state(dsa=2))                                    # another geometry: another store key, no hit
    assert other.store.key != eng.store.key
    assert disk_points.restore(other, ids + [5], 0) is None


def test_lookup_prefers_kept_snapshot(env):
    eng = _Eng(_state())
    ids = list(range(1, 900))
    eng.live = list(ids)
    disk_points.capture(eng, _snap(eng.e.st, ids))
    _wait(eng.store, eng.store.prefix(ids).point(899))
    assert disk_points.restore(eng, ids + [3], have=899) is None   # a kept snapshot as long already covers it


def test_ps_block_is_multiple_of_pool():
    assert ps.B % disk_points.POOL == 0
