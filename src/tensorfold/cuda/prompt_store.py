"""Kept prompt points on disk (local patch 0006): a prompt state outlives its GPU slot, a model switch and a restart.

One store serves every model; the budgets are shared. Under ``TF_PROMPT_CACHE_DIR``:

  index.sqlite             points, blocks, and which point uses which block
  <model key>/b/<h>.pt     1024 cache rows of every attention layer, content-addressed: shared by every point whose
                           prompt starts with the same tokens (a conversation's points write each block once)
  <model key>/p/<h>.pt     a point: its token ids, the rows past its last whole block, the recurrent state

A point is normal (deleted ``TTL`` hours after it was made, least recently used first past the normal budget) or
pinned: a prompt head that ``PIN_AFTER`` different continuations resumed from (a system prompt plus tools). A pinned
point has its own budget and no TTL, and it is never changed: a request that resumes from it gets a copy in a GPU slot,
and the points its prefill makes are normal ones. When the model key changes (new weights, image or KV format) the
pinned points of the old key are prefilled again while the server is idle, and the old copies go.

The model key names everything the saved rows depend on, so a changed engine never reads another one's rows.

Storage tiers: ``Tier`` is the place files live. The local SSD directory is the only tier the engine reads or writes;
a slower one is a mirror kept by a separate job, never read on a request's path.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import queue
import sqlite3
import threading
import time
from collections import Counter
import urllib.request
from pathlib import Path

import numpy as np
import torch

G = 256                  # tokens a step of the prefix hash chain covers (= markers.MIN_GAP, the closest two points get)
B = 1024                 # rows a cache block holds (a multiple of G and of every attention pooling ratio)
GIB = 1024**3
PACE_S = float(os.environ.get("TF_PROMPT_CACHE_PACE_S", "600") or 0)   # the writer waits for an idle engine this long
RELEASE_PINNED = os.environ.get("TF_PROMPT_CACHE_RELEASE_PINNED", "1") != "0" and hasattr(torch._C, "_host_emptyCache")


def _env(name: str, default, cast=float):
    v = os.environ.get(name, "")
    return cast(v) if v != "" else default


def _log(msg: str) -> None:
    print(f"[prompt-cache] {msg}", flush=True)


class Prefix:
    """One prompt's prefix hashes: a chain over whole G-token pieces, then a point's or a block's key in O(G)."""

    def __init__(self, seed: bytes, ids) -> None:
        self.ids = np.asarray(ids, dtype=np.int32)
        chain = [seed]
        for k in range(len(self.ids) // G):
            chain.append(hashlib.blake2b(chain[-1] + self.ids[k * G:(k + 1) * G].tobytes(), digest_size=16).digest())
        self.chain = chain

    def point(self, n: int) -> str:
        k = n // G
        return hashlib.blake2b(self.chain[k] + self.ids[k * G:n].tobytes() + int(n).to_bytes(8, "little"),
                               digest_size=16).hexdigest()

    def block(self, j: int) -> str:
        """Rows [jB, (j+1)B): a main row depends on the tokens up to it, an MTP row on the next one too."""

        end = (j + 1) * B
        return hashlib.blake2b(b"block" + self.chain[end // G] + self.ids[end:end + 1].tobytes(),
                               digest_size=16).hexdigest()


def ids_hash(ids) -> str:
    """A prompt head's hash without the model key: finds the pinned points to prefill again under a new key."""

    return hashlib.blake2b(np.asarray(ids, dtype=np.int32).tobytes(), digest_size=16).hexdigest()


class Copier:
    """A point's device tensors to pinned host memory without stopping the GPU (local patch 0008): each is first
    snapshotted on the compute stream (later writes to its source cannot race the copy), then copied on a side stream;
    ``done()`` is the event the writer waits on before it saves them."""

    def __init__(self) -> None:
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None

    def host(self, t: torch.Tensor, *, private: bool = False) -> torch.Tensor:
        """``t`` on the host (``private``: ``t`` is already a fresh tensor nothing else writes)."""

        if self.stream is None or not t.is_cuda:
            return t.detach().cpu()
        if not private:
            t = t.clone()
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            h = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
            h.copy_(t, non_blocking=True)
        t.record_stream(self.stream)                    # its memory is not reused before the copy ends
        return h

    def done(self):
        if self.stream is None:
            return None
        event = torch.cuda.Event()
        event.record(self.stream)
        return event


class Tier:
    """Where the store's files live: ``has``, ``get`` (a local path to read), ``put`` (an object), ``evict``."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def has(self, rel: str) -> bool:
        return (self.root / rel).is_file()

    def get(self, rel: str) -> Path:
        return self.root / rel

    def put(self, rel: str, obj) -> int:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save(obj, tmp)
        os.replace(tmp, path)
        return path.stat().st_size

    def evict(self, rel: str) -> None:
        try:
            (self.root / rel).unlink()
        except FileNotFoundError:
            pass


SCHEMA = """
CREATE TABLE IF NOT EXISTS points(model TEXT, key TEXT, label TEXT, n INT, ids_hash TEXT, bytes INT, total INT,
    created REAL, used REAL, hits INT DEFAULT 0, conts TEXT DEFAULT '[]', next INT, pinned INT DEFAULT 0,
    PRIMARY KEY(model, key));
CREATE TABLE IF NOT EXISTS blocks(model TEXT, hash TEXT, bytes INT, PRIMARY KEY(model, hash));
CREATE TABLE IF NOT EXISTS uses(model TEXT, point TEXT, block TEXT, PRIMARY KEY(model, point, block));
CREATE INDEX IF NOT EXISTS uses_block ON uses(model, block);
CREATE INDEX IF NOT EXISTS points_ids ON points(ids_hash);
"""


class Store:
    """The points of one model key: an in-memory index for lookups, a writer thread for files and the database."""

    def __init__(self, root: str | Path, key: str, label: str, *, idle=None) -> None:
        self.root = Path(root)
        self.key = hashlib.blake2b(key.encode(), digest_size=8).hexdigest()
        self.label = label
        self.seed = hashlib.blake2b(key.encode(), digest_size=16).digest()
        self.tier = Tier(self.root)
        self.ttl = _env("TF_PROMPT_CACHE_TTL_H", 72.0) * 3600
        self.normal_cap = int(_env("TF_PROMPT_CACHE_GIB", 100.0) * GIB)
        self.pinned_cap = int(_env("TF_PROMPT_CACHE_PINNED_GIB", 50.0) * GIB)
        self.pin_after = int(_env("TF_PROMPT_CACHE_PIN_AFTER", 3, int))
        # host memory the copies waiting for the writer may hold; past it a point waits in its GPU slot (patch 0010)
        self.queue_cap = int(_env("TF_PROMPT_CACHE_QUEUE_GIB", 4.0) * GIB)
        # patch 0011: GPU memory the points waiting in their slots may hold (their snapshots); past it, or past the
        # memory gate, the engine waits for the writer (``wait``) instead of keeping one more. Never drops a point.
        self.defer_cap = int(_env("TF_PROMPT_CACHE_DEFER_GIB", 2.0) * GIB)
        self.wait_s = _env("TF_PROMPT_CACHE_WAIT_S", 120.0)              # the longest a wait blocks; then past the cap
        self.gate = None                                                  # the engine's MemoryGate (or the host's)
        self.waiters = 0                                                  # engine threads in ``wait``: no pacing
        self.exit_s = _env("TF_PROMPT_CACHE_EXIT_S", 20.0)              # how long a stopping server waits for the writer
        self.prefill_tps = _env("TF_PROMPT_CACHE_PREFILL_TPS", 1800.0)    # tokens a second a load saves recomputing
        self.read_bps = _env("TF_PROMPT_CACHE_READ_GBS", 2.0) * 1e9        # measured from the loads as they happen
        self.lock = threading.Lock()
        self.points: dict[str, tuple[int, int]] = {}      # written points of this key -> (tokens, bytes to load)
        self.lengths: list[int] = []                      # their distinct token counts, longest first
        self.blocks: set[str] = set()
        self.pending: set[str] = set()                    # points and blocks queued, not yet written
        self.held: Counter = Counter()                    # blocks a queued point uses (not deleted meanwhile)
        self.repin: dict[str, tuple[str, str]] = {}       # ids hash -> (model, key) of an old key's pinned point
        self.queued = 0
        self.deferred: dict[str, int] = {}               # points waiting in their GPU slot for room in the queue
        self.defer_bytes: dict[str, int] = {}            # patch 0011: the GPU bytes each of them holds
        self.inflight: dict[str, int] = {}               # points queued for the writer, not yet written
        # patch 0010: no point is dropped. deferred: waited in GPU memory for the queue; forced: copied past the queue's
        # cap because its slot was needed; lost: could not be saved (a write failed, or the server stopped before the
        # writer got to it); lost_before: queued by the previous run of this key and never written (killed, power)
        self.stats = dict(saved=0, saved_bytes=0, dropped=0, loads=0, load_tokens=0, load_s=0.0, evicted=0,
                          pinned=0, errors=0, deferred=0, forced=0, lost=0, lost_before=0, queue_peak=0,
                          waits=0, wait_s=0.0, defer_peak=0)
        self.q: queue.Queue = queue.Queue()
        self.copier = Copier()
        self.idle = idle
        self.root.mkdir(parents=True, exist_ok=True)
        db = self._db()
        db.executescript(SCHEMA)
        for k, n, total in db.execute("SELECT key, n, total FROM points WHERE model=?", (self.key,)):
            self.points[k] = (n, total)
        self.blocks = {h for (h,) in db.execute("SELECT hash FROM blocks WHERE model=?", (self.key,))}
        for ih, model, k in db.execute("SELECT ids_hash, model, key FROM points WHERE pinned=1 AND label=? AND model!=?",
                                       (label, self.key)):
            self.repin[ih] = (model, k)
        db.close()
        self._lengths()
        torn = 0
        for tmp in self.root.glob("*/[bp]/*.tmp"):
            tmp.unlink(missing_ok=True)
            torn += 1
        self.jlock = threading.Lock()
        self.jpath = self.root / self.key / "journal.log"
        self._recover(torn)
        self.gate = _host_gate()
        self.room = {0: 0, 1: 0}                          # patch 0011: bytes each pool had left at the last eviction
        self.grown = 0                                    # bytes written since (an eviction scan runs once past it)
        self.unpinned = 0                                 # pinned host bytes freed since torch's cache was emptied
        atexit.register(self.close)
        _log(f"{self.root} key {self.key} ({key}): {len(self.points)} points, {len(self.blocks)} blocks; normal "
             f"{self.normal_cap / GIB:g} GiB for {self.ttl / 3600:g} h, pinned {self.pinned_cap / GIB:g} GiB after "
             f"{self.pin_after} continuations; {len(self.repin)} pinned points of an older key to prefill again")
        self.thread = threading.Thread(target=self._writer, daemon=True, name="prompt-cache-writer")
        self.thread.start()
        if self.repin and idle is not None:
            threading.Thread(target=self._rebuild, daemon=True, name="prompt-cache-rebuild").start()

    # -- the engine's side (its scheduler thread) ----------------------------------------------------------------
    def prefix(self, ids) -> Prefix:
        return Prefix(self.seed, ids)

    def has_point(self, k: str) -> bool:
        return k in self.points or k in self.pending

    def has_block(self, h: str) -> bool:
        return h in self.blocks or h in self.pending

    def lookup(self, pre: Prefix, have: int, below: int) -> tuple[str, int, int] | None:
        """The longest point the prompt extends (< ``below`` tokens) worth loading over ``have`` cached tokens."""

        with self.lock:
            lengths = list(self.lengths)
            points = self.points
            for n in lengths:
                if n >= below:
                    continue
                if n <= have:
                    return None
                k = pre.point(n)
                hit = points.get(k)
                if hit is None:
                    continue
                gain = (n - have) / self.prefill_tps
                cost = hit[1] / self.read_bps
                return (k, n, hit[1]) if gain > 1.5 * cost + 0.05 else None
        return None

    def full(self) -> bool:
        """The queue holds its cap of host copies: a new point waits in its GPU slot (``defer``) until it drains."""

        return self.queued >= self.queue_cap

    def outside(self) -> int:
        """Patch 0011: memory the store holds that no cache budget counts (``MemoryGate.outside``): the points
        waiting in GPU slots and the host copies past the queue's cap."""

        return sum(dict(self.defer_bytes).values()) + max(0, self.queued - self.queue_cap)

    def may_defer(self, k: str, nbytes: int) -> bool:
        """Patch 0011: whether a point may wait in its slot holding ``nbytes`` more: under ``defer_cap`` and the gate
        (else the caller ``wait``s for the writer and queues it)."""

        extra = max(0, int(nbytes) - self.defer_bytes.get(k, 0))
        if sum(self.defer_bytes.values()) + extra > self.defer_cap:
            return False
        return self.gate is None or self.gate.fits(extra)

    def may_force(self, nbytes: int) -> bool:
        """Patch 0011: whether a copy may go past the queue's cap now (its slot is needed): the bytes past the cap
        stay within ``defer_cap`` and the gate; else the caller ``wait``s first."""

        if not self.full():
            return True
        if max(0, self.queued - self.queue_cap) + int(nbytes) > self.defer_cap:
            return False
        return self.gate is None or self.gate.fits(int(nbytes))

    def wait(self) -> float:
        """Patch 0011 backpressure: block the engine until the writer has room in the queue (it does not pace
        meanwhile), at most ``wait_s``; then the caller copies past the cap (counted ``forced``): never dropped."""

        t0 = time.monotonic()
        self.waiters += 1
        try:
            while self.full() and self.thread.is_alive() and time.monotonic() - t0 < self.wait_s:
                time.sleep(0.002)
        finally:
            self.waiters -= 1
        s = time.monotonic() - t0
        self.stats["waits"] += 1
        self.stats["wait_s"] += s
        return s

    def defer(self, k: str, n: int, nbytes: int = 0) -> None:
        if k not in self.deferred:
            self.deferred[k] = n
            self.stats["deferred"] += 1
            self._journal(f"q {k} {n}")
        self.defer_bytes[k] = int(nbytes)
        self.stats["defer_peak"] = max(self.stats["defer_peak"], sum(self.defer_bytes.values()))

    def undefer(self, k: str) -> None:
        """A waiting point left its slot's list: queued (``put``), already there, or not savable any more."""

        self.deferred.pop(k, None)
        self.defer_bytes.pop(k, None)

    def put(self, k: str, n: int, ids, point: dict, blocks: dict, order: list[str], nbytes: int, ready=None,
            forced: bool = False) -> bool:
        """Queue a point and its new blocks for the writer. Never drops one (patch 0010): the caller keeps a point in
        its GPU slot while ``full()``, and ``forced`` copies go past the cap when that slot is needed."""

        with self.lock:
            self.queued += nbytes
            self.stats["queue_peak"] = max(self.stats["queue_peak"], self.queued)
            self.pending.add(k)
            self.pending.update(blocks)
            self.held.update(order)
            self.inflight[k] = n
        self.stats["forced"] += int(forced)
        self.defer_bytes.pop(k, None)
        if self.deferred.pop(k, None) is None:
            self._journal(f"q {k} {n}")
        self.q.put(("put", k, n, ids_hash(ids), point, blocks, order, nbytes, ready))
        return True

    def read(self, k: str) -> tuple[dict, list[dict], float]:
        t0 = time.perf_counter()
        point = torch.load(self.tier.get(f"{self.key}/p/{k}.pt"), map_location="cpu", weights_only=True)
        blocks = [torch.load(self.tier.get(f"{self.key}/b/{h}.pt"), map_location="cpu", weights_only=True)
                  for h in point["blocks"]]
        return point, blocks, t0

    def loaded(self, k: str, n: int, nbytes: int, t0: float) -> None:
        s = max(time.perf_counter() - t0, 1e-3)
        self.read_bps = 0.7 * self.read_bps + 0.3 * nbytes / s
        self.stats["loads"] += 1
        self.stats["load_tokens"] += n
        self.stats["load_s"] += s
        _log(f"loaded {n} tokens from disk in {s:.2f} s ({nbytes / s / 1e9:.1f} GB/s)")

    def used(self, pre: Prefix, n: int) -> None:
        """A request resumed from the point at ``n`` (from GPU memory or disk): its use and continuation count."""

        k = pre.point(n)
        if k in self.points:
            cont = hashlib.blake2b(pre.ids[n:n + 64].tobytes(), digest_size=8).hexdigest()
            nxt = int(pre.ids[n]) if n < len(pre.ids) else None
            self.q.put(("used", k, cont, nxt))

    # -- the writer thread ---------------------------------------------------------------------------------------
    def _db(self) -> sqlite3.Connection:
        # the default rollback journal, not WAL: the router reads this file from a read-only mount as another user
        return sqlite3.connect(self.root / "index.sqlite", timeout=60, isolation_level=None)

    def _pace(self, nbytes: int) -> None:
        """Write while the engine is idle: a big point's save (serialization and file writes in this process) slows a
        prefill running meanwhile. Never drops: it waits at most PACE_S, and not at all once the queue holds its cap
        (the engine then needs room), so a point is always written."""

        if self.idle is None or PACE_S <= 0:
            return
        t0 = time.monotonic()
        while (not self.idle() and self.queued < max(self.queue_cap, nbytes + 1) and not self.waiters
               and time.monotonic() - t0 < PACE_S):   # (a point bigger than the cap: alone in the queue)
            time.sleep(0.05)

    def _lengths(self) -> None:
        self.lengths = sorted({n for n, _ in self.points.values()}, reverse=True)

    def _writer(self) -> None:
        db = self._db()
        last = 0.0
        while True:
            try:
                op = self.q.get(timeout=60)
            except queue.Empty:
                op = None
            try:
                if op is not None and op[0] == "put":
                    self._pace(op[7])
                    try:
                        if op[-1] is not None:
                            op[-1].synchronize()              # the host copies have landed (``Copier``)
                        self._write(db, *op[1:-1])
                        self._journal(f"w {op[1]}")
                        self.unpinned += op[7]
                        # drained, or a queue's worth freed while it never drains (a long prefill's points; patch
                        # 0011): the pinned copies' memory back to the system (torch keeps freed pinned blocks
                        # cached; on unified memory that is GPU memory too)
                        if RELEASE_PINNED and (self.queued <= 0 or self.unpinned >= self.queue_cap):
                            op = ("put",)
                            self.unpinned = 0
                            torch._C._host_emptyCache()
                    except Exception as exc:  # noqa: BLE001
                        self.stats["lost"] += 1
                        self._journal(f"x {op[1]}")
                        _log(f"point of {op[2]} tokens NOT saved: {type(exc).__name__}: {exc}")
                        raise
                if op is None:
                    self._journal(None, unless_busy=True)     # nothing in flight: start the journal again
                elif op is not None and op[0] == "used":
                    self._use(db, *op[1:])
                # patch 0011: a scan after a put only once the bytes written since the last one may pass a cap
                if time.time() - last > 60 or (op is not None and op[0] == "put" and
                                               self.grown >= min(self.room.values())):
                    self._evict(db)
                    if time.time() - last > 600:
                        self._report()
                    last = time.time()
            except Exception as exc:  # noqa: BLE001 - the cache never stops the server
                self.stats["errors"] += 1
                _log(f"error: {type(exc).__name__}: {exc}")

    def _write(self, db, k, n, ih, point, blocks, order, nbytes) -> None:
        try:
            sizes = {}
            for h, obj in blocks.items():
                rel = f"{self.key}/b/{h}.pt"
                sizes[h] = self.tier.put(rel, obj) if not self.tier.has(rel) else self.tier.get(rel).stat().st_size
            size = self.tier.put(f"{self.key}/p/{k}.pt", point)
            for h in order:
                if h not in sizes:
                    row = db.execute("SELECT bytes FROM blocks WHERE model=? AND hash=?", (self.key, h)).fetchone()
                    sizes[h] = row[0] if row else 0
            now = time.time()
            old = self.repin.pop(ih, None)
            db.execute("BEGIN")
            db.executemany("INSERT OR IGNORE INTO blocks VALUES (?,?,?)", [(self.key, h, sizes[h]) for h in blocks])
            db.execute("INSERT OR REPLACE INTO points(model,key,label,n,ids_hash,bytes,total,created,used,pinned) "
                       "VALUES (?,?,?,?,?,?,?,?,?,?)",
                       (self.key, k, self.label, n, ih, size, size + sum(sizes[h] for h in order), now, now,
                        int(old is not None)))
            db.executemany("INSERT OR IGNORE INTO uses VALUES (?,?,?)", [(self.key, k, h) for h in order])
            db.execute("COMMIT")
            if old is not None:
                self._drop(db, [old])
                _log(f"pinned point of {n} tokens prefilled again under this key")
            with self.lock:
                self.points[k] = (n, size + sum(sizes[h] for h in order))
                self.blocks.update(blocks)
                self._lengths()
            self.stats["saved"] += 1
            self.stats["saved_bytes"] += size + sum(sizes[h] for h in blocks)
            self.grown += size + sum(sizes[h] for h in blocks)
        finally:
            with self.lock:
                self.inflight.pop(k, None)
                self.pending.discard(k)
                self.pending.difference_update(blocks)
                self.held.subtract(order)
                self.held += Counter()                    # drop the zero counts
                self.queued -= nbytes

    def _use(self, db, k, cont, nxt) -> None:
        row = db.execute("SELECT conts, pinned FROM points WHERE model=? AND key=?", (self.key, k)).fetchone()
        if row is None:
            return
        conts = json.loads(row[0])
        if cont not in conts and len(conts) < 16:
            conts.append(cont)
        pin = int(row[1] or len(conts) >= self.pin_after)
        db.execute("UPDATE points SET used=?, hits=hits+1, conts=?, next=COALESCE(next, ?), pinned=? "
                   "WHERE model=? AND key=?", (time.time(), json.dumps(conts), nxt, pin, self.key, k))
        if pin and not row[1]:
            self.room[1] = min(self.room[1], 0)           # the pinned pool grew: scanned at the next put
            self.stats["pinned"] += 1
            _log(f"pinned a point ({len(conts)} different continuations resumed from it)")

    def _drop(self, db, rows) -> None:
        """Delete points (model, key) and then the blocks no point uses."""

        db.execute("BEGIN")
        for model, k in rows:
            db.execute("DELETE FROM points WHERE model=? AND key=?", (model, k))
            db.execute("DELETE FROM uses WHERE model=? AND point=?", (model, k))
        orphans = db.execute("SELECT model, hash FROM blocks b WHERE NOT EXISTS "
                             "(SELECT 1 FROM uses u WHERE u.model=b.model AND u.block=b.hash)").fetchall()
        with self.lock:
            orphans = [(m, h) for m, h in orphans if not (m == self.key and (h in self.pending or self.held[h]))]
        db.executemany("DELETE FROM blocks WHERE model=? AND hash=?", orphans)
        db.execute("COMMIT")
        with self.lock:
            for model, k in rows:
                if model == self.key:
                    self.points.pop(k, None)
            for model, h in orphans:
                if model == self.key:
                    self.blocks.discard(h)
            self._lengths()
        for model, k in rows:
            self.tier.evict(f"{model}/p/{k}.pt")
        for model, h in orphans:
            self.tier.evict(f"{model}/b/{h}.pt")
        self.stats["evicted"] += len(rows)

    def _pools(self, db) -> dict[int, int]:
        """Bytes of each pool (every model's): its points, and the blocks a pinned point uses (pinned) or not (normal).
        Patch 0011: the held blocks once, by a join (a correlated EXISTS scanned every use per block: ~1 s a call on
        a full index, twice a pool after every put, so the writer fell behind a long prefill's points)."""

        normal, pinned, total, held = db.execute(
            "SELECT (SELECT COALESCE(SUM(bytes),0) FROM points WHERE pinned=0), "
            "(SELECT COALESCE(SUM(bytes),0) FROM points WHERE pinned=1), (SELECT COALESCE(SUM(bytes),0) FROM blocks), "
            "(SELECT COALESCE(SUM(b.bytes),0) FROM blocks b JOIN (SELECT DISTINCT u.model, u.block FROM points p "
            "JOIN uses u ON u.model=p.model AND u.point=p.key WHERE p.pinned=1) x "
            "ON x.model=b.model AND x.block=b.hash)").fetchone()
        return {0: normal + total - held, 1: pinned + held}

    def _pool(self, db, pinned: int) -> int:
        return self._pools(db)[pinned]

    def _evict(self, db) -> None:
        old = db.execute("SELECT model, key FROM points WHERE pinned=0 AND created<?",
                         (time.time() - self.ttl,)).fetchall()
        if old:
            self._drop(db, old)
        self.grown = 0
        caps = {0: self.normal_cap, 1: self.pinned_cap}
        pools = self._pools(db)
        for pinned, cap in caps.items():
            while pools[pinned] > cap:
                rows = db.execute("SELECT model, key FROM points WHERE pinned=? ORDER BY used LIMIT 8",
                                  (pinned,)).fetchall()
                if not rows:
                    break
                self._drop(db, rows)
                pools = self._pools(db)
        self.room = {p: caps[p] - pools[p] for p in caps}

    def _report(self) -> None:
        s = self.stats
        _log(f"{len(self.points)} points; saved {s['saved']} ({s['saved_bytes'] / GIB:.1f} GiB), loaded {s['loads']} "
             f"({s['load_tokens']} tokens, {s['load_s']:.1f} s), evicted {s['evicted']}, pinned {s['pinned']}, "
             f"errors {s['errors']}; waited in GPU {s['deferred']} (peak {s['defer_peak'] / GIB:.2f} GiB), forced "
             f"{s['forced']}, engine waited {s['waits']} ({s['wait_s']:.1f} s), queue peak "
             f"{s['queue_peak'] / GIB:.2f} GiB, LOST {s['lost']} (previous run {s['lost_before']})")
        try:
            tmp = self.root / self.key / "stats.json.tmp"
            tmp.write_text(json.dumps({**s, "label": self.label, "points": len(self.points), "time": time.time()}))
            os.replace(tmp, self.root / self.key / "stats.json")
        except OSError:
            pass

    # -- never lose a point (patch 0010) ----------------------------------------------------------------------------
    def _journal(self, line: str | None, unless_busy: bool = False) -> None:
        """One line per point queued (q), written (w) or failed (x); ``None`` empties it. Survives a kill: the next
        start counts the queued points never written."""

        try:
            with self.jlock:
                if unless_busy and (self.inflight or self.deferred):
                    return
                self.jpath.parent.mkdir(parents=True, exist_ok=True)
                with open(self.jpath, "w" if line is None else "a") as f:
                    if line is not None:
                        f.write(line + "\n")
        except OSError:
            pass

    def _recover(self, torn: int) -> None:
        if not self.jpath.is_file():
            return
        queued: dict[str, str] = {}
        for line in self.jpath.read_text(errors="ignore").splitlines():
            part = line.split()
            if len(part) >= 2 and part[0] == "q":
                queued[part[1]] = part[2] if len(part) > 2 else "?"
            elif len(part) >= 2:
                queued.pop(part[1], None)
        lost = {k: n for k, n in queued.items() if k not in self.points}
        self.stats["lost_before"] = len(lost)
        if lost or torn:
            _log(f"LOST at the previous stop (killed before the writer got to them): {len(lost)} points "
                 f"({', '.join(sorted(lost.values(), key=lambda x: int(x) if x.isdigit() else 0))} tokens); "
                 f"{torn} torn files removed")
        self._journal(None)

    def close(self) -> None:
        """At exit (a stop is SIGTERM, then 30 s): give the writer up to ``TF_PROMPT_CACHE_EXIT_S`` to finish."""

        t0 = time.time()
        while (self.inflight or not self.q.empty()) and time.time() - t0 < self.exit_s:
            time.sleep(0.1)
        left = len(self.inflight) + len(self.deferred)
        if left:
            self.stats["lost"] += left
            _log(f"stopping with {left} points not saved (counted as lost at the next start unless written)")
        self._report()

    # -- pinned points of an older key ----------------------------------------------------------------------------
    def _rebuild(self) -> None:
        """Prefill each old key's pinned point again (through this server, one at a time, only while it is idle)."""

        port, model = os.environ.get("PORT", "8001"), os.environ.get("SERVED_NAME", self.label)
        base = f"http://127.0.0.1:{port}"
        while True:                                           # until the server answers
            try:
                urllib.request.urlopen(base + "/v1/models", timeout=5).read()
                break
            except Exception:  # noqa: BLE001
                time.sleep(5)
        db = self._db()
        todo = db.execute("SELECT model, key, next FROM points WHERE pinned=1 AND label=? AND model!=? "
                          "ORDER BY used DESC", (self.label, self.key)).fetchall()
        db.close()
        for model_key, k, nxt in todo:
            if nxt is None:
                continue
            quiet = 0
            while quiet < 30:                                 # 30 s without a request
                time.sleep(1)
                quiet = quiet + 1 if self.idle() else 0
            try:
                ids = torch.load(self.tier.get(f"{model_key}/p/{k}.pt"), map_location="cpu",
                                 weights_only=True)["ids"].tolist()
                body = json.dumps({"model": model, "prompt": ids + [int(nxt)], "max_tokens": 1,
                                   "temperature": 0}).encode()
                urllib.request.urlopen(urllib.request.Request(base + "/v1/completions", body,
                                                              {"Content-Type": "application/json"}),
                                       timeout=3600).read()
            except Exception as exc:  # noqa: BLE001
                _log(f"prefilling a pinned point again failed: {type(exc).__name__}: {exc}")


def _host_gate():
    """Patch 0011: a gate on live memory for a store whose engine has none (``Store.gate``; an engine with its own
    MemoryGate replaces it): the driver's free memory, MemAvailable on a unified GPU, less TF_PROMPT_CACHE_RESERVE_GIB."""

    if not torch.cuda.is_available():
        return None
    from .capacity import available_bytes
    from .memory_gate import MemoryGate, torch_live

    return MemoryGate(1 << 62, int(_env("TF_PROMPT_CACHE_RESERVE_GIB", 2.0) * GIB), torch_live(torch, available_bytes))


def open_store(geometry: str, *, idle=None) -> Store | None:
    """The store named by the environment (``TF_PROMPT_CACHE_DIR``), or None when it is off."""

    root = os.environ.get("TF_PROMPT_CACHE_DIR", "")
    if not root:
        return None
    key = f"{os.environ.get('TF_PROMPT_CACHE_KEY', '')}|{geometry}"
    label = os.environ.get("SERVED_NAME", "") or os.environ.get("TF_PROMPT_CACHE_KEY", "")
    try:
        return Store(root, key, label, idle=idle)
    except Exception as exc:  # noqa: BLE001
        _log(f"off: {type(exc).__name__}: {exc}")
        return None
