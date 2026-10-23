"""Qwen3.6 MoE's concurrent rounds on one GPU: every stream commits exactly its own path, so it equals its serial decoding."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import torch

from tensorfold.cuda.markers import MIN_GAP
from tensorfold.cuda.sampling import sample_rows, sample_streams
from tensorfold.cuda.streams import PrefixCache, Stream, accept, next_fill
from tensorfold.engine.grammar import GrammarError
from tensorfold.families.qwen3_5.cuda.decode import CopyIndex
from tensorfold.families.qwen3_5.cuda.engine import entry_end
from tensorfold.families.qwen3_5.cuda.forward import State, _mm, commit_streams, multi_tree_forward, replay_pending
from tensorfold.families.qwen3_5.cuda.multi import kept, private

from tensorfold.cuda import keep_rank

from . import disk_points, policy_for
from .decode import COPY_ROWS, Carry, extend, mtp_round, picks
from .mtp import Cache, Head

STEP = int(os.environ.get("TF_PREFILL_STEP", "1024"))   # prompt rows a prefill step takes while other streams decode
# a round's accepted DeltaNet rows are stepped into the states by the next round's tree kernel (no replay pass)
DEFER = os.environ.get("TF_QWEN36_GDN_DEFER", "1") != "0"


# patch 0011: the most a prefill step's points hold before the first token (TF_PROMPT_CACHE_DEFER_GIB with the store)
LATER_CAP = int(float(os.environ.get("TF_PROMPT_CACHE_DEFER_GIB", "") or 2.0) * 1024**3)


class KeptEnds(PrefixCache):
    """``PrefixCache`` whose cap drops by ``keep_rank`` (patch 0010; 0103 before): a state inside a conversation's
    history first (a newer state of it extends it; it is on disk), then a conversation's newest state, then a shared
    starting point (a reused system prompt, a point resumed with 2+ different continuations, the base of 2+ diverging
    kept states), the oldest first within each (a hit makes an entry the newest); a state not yet copied for the disk
    store last of all."""

    def __init__(self, keep: int = 8) -> None:
        super().__init__(keep)
        self.conts: dict = {}                              # point -> the continuations requests resumed it with
        self.waiting = lambda entry: False                 # not yet copied for the disk store (set by the decoder)

    def _drop(self, among: list) -> None:
        idx = {id(e): i for i, e in enumerate(self.entries)}
        at = keep_rank.victim([e[0] for e in self.entries],
                              [len(self.conts.get(keep_rank.point_key(e[0]), ())) for e in self.entries],
                              [self.waiting(e) for e in self.entries], [idx[id(e)] for e in among if id(e) in idx])
        gone = self.entries[at]
        self.entries = [e for e in self.entries if e is not gone]
        self.hit &= {tuple(e[0]) for e in self.entries}


@dataclass
class Drafts:
    """A drafting stream's MTP head: its attention cache, and the committed rows it has not absorbed (``carry``)."""

    cache: Cache
    carry: Carry


def own(mc: Cache, rows: int) -> Cache:
    """A head cache of ``rows`` slots of its own, the absorbed rows of ``mc`` (a kept one's) copied in."""

    other = object.__new__(Cache)
    other.k, other.v = mc.k.new_empty((rows, *mc.k.shape[1:])), mc.v.new_empty((rows, *mc.v.shape[1:]))
    other.k[:mc.pos], other.v[:mc.pos] = mc.k[:mc.pos], mc.v[:mc.pos]
    other.pos = mc.pos
    return other


class MultiDecoder:
    """The ``Scheduler``'s decoder: a prefill step, then every stream's window (16 rows at most) in one forward; a lone stream replays ``graphs``."""

    store = None                                     # the prompt store on disk (patch 0104); set by __init__ when it is on
    unsaved = ()                                     # patch 0010/0107: kept states waiting for room in the disk queue

    def __init__(self, w, head: Head | None, *, depth: int, confidence: float, context: int = 0, keep: int = 3,
                 points=None, graphs=None) -> None:
        if head is not None and depth > 0 and not 1 <= depth <= 15:
            raise ValueError(f"MTP drafts a round with --parallel: 1 to 15 (a window holds 16 rows), not {depth}")
        self.w, self.head = w, head if depth > 0 else None
        self.depth, self.confidence = int(depth), float(confidence)
        self.context = context                       # prompt, reply and draft slots a stream holds (0: no bound)
        self.eos = tuple(w.config.eos)
        self.ids = head.ids.cpu().numpy() if self.head is not None and head.ids is not None else None
        self.streams: dict[int, Stream] = {}         # decoding
        self.filling: list[Stream] = []              # admitted, prompts still prefilling (oldest first)
        self.points = points                         # a prompt's message starts to keep states at, or None
        self.cache = KeptEnds(keep)                  # (ids, state, (head cache, held row)) at message starts and ends
        self.next_id = 0
        self.graphs = graphs if self.head is not None else None
        self.resident: Stream | None = None          # the stream whose state is in the graphs' buffers
        self.store = disk_points.open_store(self) if self.head is not None else None
        self.unsaved = []
        self._watch()

    def _watch(self) -> None:
        self.cache.waiting = lambda e: any(u[2] is e[1] for u in self.unsaved)

    def live(self) -> int:
        return len(self.streams) + len(self.filling)

    @torch.no_grad()
    def admit(self, s: Stream) -> None:
        """Queue a request on the longest kept prefix of its prompt, its caches sized once; rounds prefill the rest."""

        if self.context:
            room = self.context - len(s.prompt) - self.depth - 1
            if room < 1:
                raise ValueError(f"a prompt of {len(s.prompt)} tokens leaves no room in the {self.context}-token "
                                 "context (--context)")
            s.count = min(s.count, room)
        drafting = s.draft and self.head is not None
        if drafting and self.store is not None:
            disk_points.restore(self, s.prompt)
        hit = self.cache.longest(s.prompt) if drafting else None
        if hit is not None:
            keep_rank.note(self.cache.conts, s.prompt, len(hit[0]), [e[0] for e in self.cache.entries])
        if hit is not None and self.store is not None:
            self.store.used(self.store.prefix(s.prompt), len(hit[0]))
        need = len(s.prompt) + s.count                # the most the stream's attention caches ever hold
        s.st = private(hit[1] if hit else State(self.w), need)
        s.snap = None
        if drafting:                                  # the head writes up to depth - 1 draft slots past the committed
            cache = own(hit[2][0], need + self.depth) if hit else Cache(self.w, need + self.depth)
            s.snap = Drafts(cache, Carry(hit[2][1] if hit else None, []))
        s.sid, s.cached = self.next_id, len(hit[0]) if hit else 0
        self.next_id += 1
        s.stops = ([p for p in self.points(s.prompt) if p >= s.st.pos + MIN_GAP]
                   if self.points is not None and drafting else [])
        self.filling.append(s)

    def _fill(self) -> list[Stream]:
        """Prefill the oldest queued prompt a step: to its next kept state, or STEP rows while others decode."""

        s = next_fill(self.filling)
        pos, n = s.st.pos, len(s.prompt)
        stop = n                                 # patch 0126: points are kept inside the step's passes, never cut one
        if s.background or any(not x.done for x in self.streams.values()):   # a later foreground prompt waits one step
            stop = min(stop, pos + STEP)
        try:
            first = self._step(s, stop)
        except Exception as exc:                 # noqa: BLE001  (this request fails, the others go on)
            s.later = []
            self.filling = [x for x in self.filling if x is not s]
            s.error, s.done = exc, True
            return [s]
        if first is None:
            self._flush_points(s)
            return []
        s.take([first], self._ends(s))
        self._flush_points(s)                    # the points' disk copies after the first token, not before it
        return [s] if s.done else []

    def _flush_points(self, s: Stream) -> None:
        for args in getattr(s, "later", ()):
            self._add(*args)
        s.later = []

    def _keep(self, ids: list[int], s: Stream) -> None:
        d = s.snap
        self._add(ids, kept(s.st), (d.cache.view(), d.carry.states.clone()))

    def _add(self, ids: list[int], state, snap) -> None:
        """A kept entry, and its copy queued for the disk store when that is on."""

        if self.store is not None and snap[1] is not None:
            try:
                disk_points.capture(self, ids, state, snap)
            except Exception as exc:  # noqa: BLE001 - the disk copy never stops a request
                self.store.stats["errors"] += 1
                print(f"[prompt-cache] save failed: {type(exc).__name__}: {exc}", flush=True)
        self.cache.add(ids, state, snap)

    def _step(self, s: Stream, stop: int) -> int | None:
        """Prefill prompt[pos:stop] (the same bits for any stops); at the end, sample the first token and start decoding."""

        t0 = time.perf_counter()
        d: Drafts | None = s.snap
        end = (entry_end(s.prompt) if d is not None and not
               (s.stops and len(s.prompt) - s.stops[-1] < MIN_GAP) else None)

        s.later = []

        cap = self.store.defer_cap if self.store is not None else LATER_CAP

        def keep(point, state, cache, held):     # added (and queued for the disk) once the step has sampled
            s.later.append((list(s.prompt[:point]), kept(state), (cache.view(), held.clone())))
            # patch 0011: a long prompt's points in one step hold a DeltaNet copy each: past the cap the oldest are
            # added (and queued) now, so a step's points never hold more than that
            while len(s.later) > 1 and sum(disk_points.held_bytes(x[1], x[2]) for x in s.later) > cap:
                self._add(*s.later.pop(0))

        try:
            normed, held = extend(self.w, self.head if d is not None else None, s.prompt, s.st,
                                  d.cache if d is not None else None, d.carry.states if d is not None else None, stop,
                                  keep_at=end, keep=keep if end is not None or s.stops else None,
                                  points=[p for p in s.stops if s.st.pos < p <= stop])
            if d is not None:
                d.carry = Carry(held, [])
            first = None
            if stop == len(s.prompt):
                logits = _mm(normed[-1:], self.w.head)
                if s.constraint is not None:          # the first token under the reply's grammar
                    logits = s.constraint.mask(logits)
                first = sample_rows(logits, [len(s.prompt)], s.sampling)[0]
                if s.constraint is not None:
                    s.constraint.advance([first])
        finally:
            s.prefill_s += time.perf_counter() - t0
        if first is None:
            return None
        if d is not None:
            d.carry.tokens = [first]
        s.copies = CopyIndex() if d is not None else None
        s.context = list(s.prompt)
        s.started = time.perf_counter()
        self.filling = [x for x in self.filling if x is not s]
        self.streams[s.sid] = s                       # after every step that can fail: a failure leaves it queued
        return first

    @torch.no_grad()
    def round(self) -> list[Stream]:
        """A prefill step for the next queued prompt, then one round over the decoding streams; returns the finished."""

        if self.unsaved:
            disk_points.retry(self)                  # patch 0107: as soon as the disk queue has room
        done = self._fill() if self.filling else []
        live = [s for s in self.streams.values() if not s.done]
        if not live:
            return done
        if len(live) == 1 and self._fits(live[0]):
            return done + self._alone(live[0])
        self._propose(live)
        grammars = self._constrain(live)
        done += [s for s in live if s.done]
        live = [s for s in live if not s.done]
        if not live:
            return done
        wins = [[s.out[-1]] + s.drafts for s in live]
        chains = [list(range(-1, len(t) - 1)) for t in wins]
        logits, record, hidden, starts = multi_tree_forward(
            self.w, [(t, p, s.st) for t, p, s in zip(wins, chains, live)], hidden=True, pending=self._pending(live))
        for k, s in enumerate(live):                  # a constrained stream's rows, each masked by its path
            if s.sid in grammars:
                s.constraint.mask(logits[starts[k]:starts[k + 1]], grammars[s.sid])
        positions = [[s.st.pos + 1 + i for i in range(len(t))] for s, t in zip(live, wins)]
        sampled = sample_streams(logits, starts, positions, [s.sampling for s in live])
        kept_rows = [accept(t, p, rows, s.count - len(s.out), self._ends(s))
                     for s, t, p, rows in zip(live, wins, chains, sampled)]
        paths = [[starts[k] + r for r in path] for k, (path, _) in enumerate(kept_rows)]
        commit_streams([s.st for s in live], record, paths, in_place=True, defer=DEFER)
        if DEFER:
            self.last_record = record
            for s, path in zip(live, paths):
                s.pend = (record, path)
        for k, (s, tokens, (path, end)) in enumerate(zip(live, wins, kept_rows)):
            new = [tokens[r] for r in path[1:]] + [end]
            s.committed.extend(tokens[r] for r in path)
            s.counted(len(tokens))
            if s.snap is not None:                    # the kept rows' final states, for the head to absorb next
                s.snap.carry = Carry(hidden[starts[k] + path[0]:starts[k] + path[-1] + 1], new)
            if s.constraint is not None and not self._follow(s, new):
                continue
            s.take(new, self._ends(s))
        return done + [s for s in live if s.done]

    def _constrain(self, live: list[Stream]) -> dict:
        """Each constrained stream's drafts cut where its grammar rejects one, and its rows' masks by stream id."""

        grammars = {}
        for s in live:
            if s.constraint is None:
                continue
            try:
                window = s.constraint.window([s.out[-1]] + s.drafts, list(range(-1, len(s.drafts))))
            except GrammarError as exc:               # this request ends with its error, the others go on
                s.error, s.done = exc, True
                continue
            s.drafts, grammars[s.sid] = window.tokens[1:], window
        return grammars

    @staticmethod
    def _follow(s: Stream, new: list[int]) -> bool:
        """The grammar follows the kept tokens; False (the stream ended with the error) when it fails."""

        try:
            s.constraint.advance(new)
        except GrammarError as exc:
            s.error, s.done = exc, True
            return False
        return True

    def _ends(self, s: Stream) -> tuple[int, ...]:
        """The end tokens that end this stream: none when its request ignores them (``ignore_eos``)."""

        return self.eos if s.stop_eos else ()

    def _fits(self, s: Stream) -> bool:
        """Whether a drafting stream's caches fit the graphs' buffers (else it decodes eagerly in its own)."""

        return (s.snap is not None and self.graphs is not None and
                (s is self.resident or len(s.prompt) + s.count + self.depth <= self.graphs.capacity))

    def _alone(self, s: Stream) -> list[Stream]:
        """The lone stream's round in the one-stream graphs, its state moved into their buffers once (eager there when others join)."""

        g, d = self.graphs, s.snap
        depth, conf, before = policy_for(1, self.depth, self.confidence)
        self._flush([s])
        if self.resident is not s:
            s.st, d.cache = g.load(s.st, d.cache, min(g.capacity, len(s.prompt) + s.count + COPY_ROWS))
            self.resident = s
        try:
            tokens, path, new, d.carry = mtp_round(s.st, d.cache, d.carry, s.out[-1], s.count - len(s.out),
                                                   s.sampling, s.context, s.copies, depth=depth,
                                                   confidence=conf, before=before, ids=self.head.ids, verify=g.verify,
                                                   step=g.draft, eos=self._ends(s), in_place=True,
                                                   constraint=s.constraint)
        except GrammarError as exc:                   # this request ends with its error
            s.error, s.done = exc, True
            return [s]
        s.committed.extend(tokens[r] for r in path)
        s.counted(len(tokens))
        s.take(new, self._ends(s))
        return [s] if s.done else []

    @staticmethod
    def _flush(streams) -> None:
        """Step the streams' deferred rows into their states now (one replay launch a record)."""

        by: dict[int, tuple] = {}
        for s in streams:
            p = getattr(s, "pend", None)
            if p is not None:
                group = by.setdefault(id(p[0]), (p[0], [], []))
                group[1].append(s.st)
                group[2].append(p[1])
                s.pend = None
        for record, states, rows in by.values():
            replay_pending(states, record, rows)

    def _pending(self, live: list[Stream]):
        """(record, rows a stream) for ``multi_tree_forward``: the deferred rows of the last forward's record; a stream
        holding an older record's rows (none in steady state) is stepped eagerly first."""

        last = getattr(self, "last_record", None)
        self._flush([s for s in live if getattr(s, "pend", None) is not None and s.pend[0] is not last])
        if not any(getattr(s, "pend", None) is not None for s in live):
            return None
        rows = []
        for s in live:
            rows.append(s.pend[1] if getattr(s, "pend", None) is not None else [])
            s.pend = None                             # the forward steps them
        return last, rows

    def _propose(self, live: list[Stream]) -> None:
        """Every drafting stream's carry absorbed in one head call, then a copy or an MTP chain each, the chains stepped together."""

        for s in live:
            s.drafts = []
        todo = [s for s in live if s.snap is not None]
        if not todo:
            return
        depth, conf, before = policy_for(len(live), self.depth, self.confidence)
        heads = [s.snap for s in todo]
        normed = self.head.forward_streams([d.cache for d in heads], [d.carry.states for d in heads],
                                           [d.carry.tokens for d in heads], [d.cache.pos for d in heads])
        last, row = [], 0
        for d in heads:
            row += len(d.carry.tokens)
            last.append(row - 1)
            d.cache.pos += len(d.carry.tokens)
        active = []
        for s, r in zip(todo, last):                  # an exact repeat of the context first: a long, likely window
            s.drafts = s.copies.propose(s.context, COPY_ROWS - 1)
            if not s.drafts:
                active.append((s, r))
        if not active:
            return
        rows = normed.index_select(0, torch.tensor([r for _, r in active], device=normed.device))
        while active:
            logits = self.head.logits(rows)
            chosen = picks(logits, [s.st.pos + 1 + len(s.drafts) for s, _ in active],
                           [s.sampling for s, _ in active], self.ids)
            going = []
            for k, ((s, _), (token, prob)) in enumerate(zip(active, chosen)):
                if before and s.drafts and prob < conf:       # ``before``: the low draft is not verified
                    continue
                s.drafts.append(token)
                if prob >= conf and len(s.drafts) < depth:
                    going.append((s, k))
            if not going:
                return
            rows = self.head.forward_streams(
                [s.snap.cache for s, _ in going], [rows[k:k + 1] for _, k in going], [[s.drafts[-1]] for s, _ in going],
                [s.snap.cache.pos + len(s.drafts) - 1 for s, _ in going])
            active = [(s, k) for k, (s, _) in enumerate(going)]

    @torch.no_grad()
    def warm(self, streams: int) -> None:
        """A synthetic request and forwards at every row count ``streams`` windows bring, so no request compiles a kernel; then forgotten."""

        s = Stream([0] * (min(300, self.context - self.depth - 2) if self.context else 300), 2, None, draft=True)
        store, self.store = self.store, None             # the warm-up prompt's states stay off the disk
        try:
            self.admit(s)
            while not s.done:
                self.finish(self.round())
            self.finish([s])
        finally:
            self.store = store
        self.cache = KeptEnds(self.cache.keep)
        self._watch()
        st = State(self.w)
        for rows in sorted({1, 16, 32, 64, 128, 16 * streams}):
            if rows > 16 * streams:
                continue
            n = -(-rows // 16)
            sizes = [rows // n + (i < rows % n) for i in range(n)]
            multi_tree_forward(self.w, [([0] * k, list(range(-1, k - 1)), st) for k in sizes], hidden=True)
            if self.head is not None:
                cache = Cache(self.w, 16)
                picks(self.head.logits(self.head.forward_streams([cache] * n, [torch.zeros(
                    (k, self.w.config.hidden), dtype=torch.bfloat16, device=self.w.norm.device) for k in sizes],
                    [[0] * k for k in sizes], [0] * n)), [1] * rows, [None] * rows, self.ids)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def finish(self, done: list[Stream]) -> None:
        """Drop finished streams (their prompt-end states joined the prefix cache at the end of their prefill)."""

        for s in done:
            self.streams.pop(s.sid, None)
            if s is self.resident:
                self.resident = None
        if self.unsaved and not self.streams and not self.filling:
            disk_points.retry(self, force=True)      # patch 0107: idle: every waiting state is copied now

    def drop(self) -> list[Stream]:
        """After an error in a round: forget the live streams and the queued prompts."""

        live = [s for s in self.streams.values() if not s.done]
        for s in live:
            del self.streams[s.sid]
        live += self.filling
        self.filling = []
        self.resident = None
        return live
