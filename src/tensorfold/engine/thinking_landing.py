"""CPU-only, token-observation state machine for bounded thinking.

The caller initializes ``initial_thinking`` from the *current assistant* prompt,
not historical prompt blocks. Before sampling each output token, check
``next_forced()``; it peeks and must be honored until the queue drains. Call
``observe`` (or this object's callable alias) only after a token is actually
emitted. Apply ``close_bias()`` to the close-token logit, mask both thinking
delimiters whenever ``closed`` is true, and honor ``should_stop`` after forced
landing. This module does not import model libraries or alter the output cap.
"""
from __future__ import annotations

from collections import Counter, deque
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, fields, replace
import json
import math
import re
import unicodedata
from typing import Callable, Iterable


def _integer(name: str, value: object, minimum: int = 0) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer, not a boolean")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


def _finite_number(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class LandingConfig:
    enabled: bool = True
    budget: int = 4000
    repeat_ngram: int = 8
    repeat_count: int = 3
    repeat_window: int = 512
    variety_window: int = 256
    variety_fraction: float = 0.25
    grace_tokens: int = 48
    answer_reserve: int = 1024
    bias_start_fraction: float = 0.5
    bias_max: float = 8.0
    repeat_min_thinking_tokens: int = 1500
    repeat_arithmetic_fraction: float = 0.75
    answer_max_tokens: int = 2048
    # False: only the budget may land thinking; repeat and low-variety are off.
    thinking_triggers: bool = True
    # False: a landed answer runs to the ordinary output cap, with no answer
    # cap and no answer-channel repeat/low-variety stop.
    answer_guard: bool = True
    # Degenerate collapse ("Hmm, hmm. Hmm, hmm."): a full variety window whose
    # non-arithmetic tokens number at least collapse_min_eligible but include at
    # most collapse_max_distinct distinct IDs. Zero disables it. It ignores
    # thinking_triggers, answer_guard and the repeat arming delay. In thinking it
    # lands and bans those IDs for collapse_ban_tokens emitted tokens; in a
    # landed answer it ends generation.
    collapse_max_distinct: int = 0
    collapse_min_eligible: int = 64
    collapse_ban_tokens: int = 256

    def __post_init__(self) -> None:
        for name in ("collapse_max_distinct", "collapse_ban_tokens"):
            _integer(name, getattr(self, name))
        _integer("collapse_min_eligible", self.collapse_min_eligible, 1)
        for name in ("enabled", "thinking_triggers", "answer_guard"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a boolean")
        for name in ("budget", "grace_tokens", "answer_reserve", "repeat_min_thinking_tokens"):
            _integer(name, getattr(self, name))
        for name in ("repeat_ngram", "repeat_count", "repeat_window", "variety_window"):
            _integer(name, getattr(self, name), 1)
        _integer("answer_max_tokens", self.answer_max_tokens, 1)
        if self.answer_max_tokens > 2048:
            raise ValueError("answer_max_tokens must not exceed 2048")
        if self.repeat_ngram > self.repeat_window:
            raise ValueError("repeat_ngram must not exceed repeat_window")
        if self.answer_reserve < 1024:
            raise ValueError("answer_reserve must be >= 1024")
        if self.grace_tokens > 48:
            raise ValueError("grace_tokens must not exceed 48")
        for name in ("variety_fraction", "bias_start_fraction", "bias_max",
                     "repeat_arithmetic_fraction"):
            _finite_number(name, getattr(self, name))
        if not 0 < self.variety_fraction <= 1:
            raise ValueError("variety_fraction must be in (0, 1]")
        if not 0 < self.repeat_arithmetic_fraction <= 1:
            raise ValueError("repeat_arithmetic_fraction must be in (0, 1]")
        if not 0 <= self.bias_start_fraction < 1:
            raise ValueError("bias_start_fraction must be in [0, 1)")
        if self.bias_max < 0:
            raise ValueError("bias_max must be >= 0")

    @classmethod
    def from_request(cls, body: Mapping, defaults: LandingConfig | None = None) -> LandingConfig:
        """Merge only thinking options; top-level thinking_budget wins, even 0.

        Unknown keys inside thinking_loop are errors. Other top-level request
        fields belong to the generation API and are deliberately ignored.
        """
        if not isinstance(body, Mapping):
            raise TypeError("request body must be a mapping")
        if defaults is not None and not isinstance(defaults, cls):
            raise TypeError("defaults must be a LandingConfig")
        options = body.get("thinking_loop", {})
        if not isinstance(options, dict):
            raise TypeError("thinking_loop must be a dict")
        unknown = set(options) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown thinking_loop fields: {sorted(map(str, unknown))}")
        # Validate the nested values even if a top-level override supersedes one.
        merged = replace(defaults or cls(), **options)
        if "thinking_budget" in body:
            merged = replace(merged, budget=body["thinking_budget"])
        return merged


class ThinkingLanding:
    """One request's thinking lifecycle, driven solely by emitted token IDs.

    ``thinking_count`` includes sampled and forced reasoning tokens, but not
    delimiters. Metadata records those two counts separately. Trigger names are
    ``repeat_ngram``, ``low_variety``, or ``budget``. ``landed`` becomes true only
    after the queued forced close is emitted, never on a natural close.

    Logging accepts a callable taking one JSON string, or a standard logger
    exposing ``info``. One landing event is logged at actual forced closure,
    including the pending header. An answer termination event records the
    first repeat, low-variety, or cap stop. Natural answers are not guarded.

    ``thinking_triggers=False`` leaves the budget as the only thinking trigger.
    ``answer_guard=False`` removes the landed-answer cap and stop entirely.
    """

    def __init__(
        self,
        config: LandingConfig,
        open_id: int,
        close_id: int,
        landing_ids: Iterable[int],
        output_cap: int,
        initial_thinking: bool = True,
        logger: Callable[[str], object] | object | None = None,
        *,
        answer_prefix_ids: Iterable[int] = (),
    ) -> None:
        if not isinstance(config, LandingConfig):
            raise TypeError("config must be a LandingConfig")
        _integer("open_id", open_id)
        _integer("close_id", close_id)
        _integer("output_cap", output_cap, 1)
        if open_id == close_id:
            raise ValueError("open_id and close_id must differ")
        if type(initial_thinking) is not bool:
            raise TypeError("initial_thinking must be a boolean")
        if logger is not None and not (callable(logger) or callable(getattr(logger, "info", None))):
            raise TypeError("logger must be callable or expose info()")
        phrase = tuple(landing_ids)
        for token in phrase:
            _integer("landing token", token)
            if token in (open_id, close_id):
                raise ValueError("landing_ids must not contain thinking delimiters")
        answer_prefix = tuple(answer_prefix_ids)
        for token in answer_prefix:
            _integer("answer prefix token", token)
            if token in (open_id, close_id):
                raise ValueError("answer_prefix_ids must not contain thinking delimiters")
        self.config = config
        self.open_id = open_id
        self.close_id = close_id
        self.landing_ids = phrase
        self.answer_prefix_ids = answer_prefix
        self.output_cap = output_cap
        self.logger = logger
        self.thinking = initial_thinking
        self.closed = False
        self.thinking_count = 0
        self.trigger: str | None = None
        self.forced: deque[int] = deque()
        self.total_emitted = 0
        self.landed = False
        self.answer_count = 0
        self.answer_trigger: str | None = None
        self.answer_trigger_count: int | None = None
        self._sampled_thinking = 0
        self._forced_thinking = 0
        self._trigger_count: int | None = None
        self._grace_used = 0
        self._banned: frozenset[int] = frozenset()
        self._ban_until = 0
        self._text_tail = ""
        self._ngram_tail: deque[int] = deque(maxlen=config.repeat_ngram)
        self._ngram_pieces: deque[str] = deque(maxlen=config.repeat_ngram)
        self._ngrams: deque[tuple[int, ...]] = deque()
        self._ngram_counts: Counter[tuple[int, ...]] = Counter()
        self._variety_tokens: deque[tuple[int, bool]] = deque()
        self._variety_counts: Counter[int] = Counter()
        self._variety_eligible = 0
        self._validate_capacity(0)
        self.effective_budget = self._budget_at(0)
        if initial_thinking:
            self._begin_thinking()

    def _required_room(self) -> int:
        # Explicit budget=0 closes an empty think block without landing prose.
        phrase_length = 0 if self.config.budget == 0 else len(self.landing_ids)
        return self.config.answer_reserve + phrase_length + 1 + len(self.answer_prefix_ids)

    def _validate_capacity(self, prefix: int) -> None:
        if self.config.enabled and self.output_cap - prefix < self._required_room():
            raise ValueError("output_cap cannot fit answer_reserve plus forced phrase, close and prefix")

    def _budget_at(self, prefix: int) -> int:
        if self.config.budget == 0:
            return 0
        room = (self.output_cap - prefix - self.config.answer_reserve
                - len(self.landing_ids) - 1 - len(self.answer_prefix_ids)
                - self.config.grace_tokens)
        return max(0, min(self.config.budget, room))

    def _begin_thinking(self) -> None:
        self._validate_capacity(self.total_emitted)
        self.effective_budget = self._budget_at(self.total_emitted)
        if self.config.enabled and self.effective_budget == 0:
            self._set_trigger("budget")
            # An explicit zero budget must not permit even one sampled token.
            # For a reserve-clamped zero, also force now, without a grace phase.
            self._queue_landing()

    def _set_trigger(self, reason: str) -> None:
        self.trigger = reason
        self._trigger_count = self.thinking_count
        self._grace_used = 0

    def _queue_landing(self) -> None:
        if not self.forced and self.thinking and not self.closed:
            if self.config.budget != 0:
                self.forced.extend(self.landing_ids)
            self.forced.append(self.close_id)
            self.forced.extend(self.answer_prefix_ids)

    def next_forced(self) -> int | None:
        """Peek at the next forced ID; only observe() advances the queue."""
        return self.forced[0] if self.forced else None

    def _sentence_boundary(self, piece: str) -> bool:
        # Newlines are unambiguous. Keep a short tail to catch digit + '.' split
        # across token boundaries, conservatively withholding on decimal-like
        # endings (including a trailing digit + '.' awaiting its next digit).
        self._text_tail = (self._text_tail + piece)[-128:]
        if "\n" in piece or "\r" in piece:
            return True
        if not piece.strip():
            return False
        stripped = self._text_tail.rstrip().rstrip('\"\'\u2019\u201d)]}')
        if not stripped or stripped[-1] not in ".!?":
            return False
        if stripped[-1] == "." and re.search(r"\d\.$", stripped):
            return False
        return True

    def _arithmetic(self, pieces: Iterable[str]) -> bool:
        """Shared Unicode classifier; unknown/blank pieces are not arithmetic."""
        total = arithmetic = 0
        for piece in pieces:
            for char in piece:
                if char.isspace():
                    continue
                total += 1
                category = unicodedata.category(char)
                if category[0] in ("N", "P") or category == "Sm" or char in "+-*/%=^<>|~":
                    arithmetic += 1
        return bool(total) and arithmetic / total >= self.config.repeat_arithmetic_fraction

    def _reset_detectors(self) -> None:
        self._text_tail = ""
        self._ngram_tail.clear()
        self._ngram_pieces.clear()
        self._ngrams.clear()
        self._ngram_counts.clear()
        self._variety_tokens.clear()
        self._variety_counts.clear()
        self._variety_eligible = 0

    def _detect(self, token: int, piece: str, *, answer: bool = False) -> str | None:
        cfg = self.config
        # Repeat and low-variety need their flag; history is kept regardless so
        # the collapse check below always sees a true rolling window.
        soft = cfg.answer_guard if answer else cfg.thinking_triggers
        armed = soft and (answer or self.thinking_count >= cfg.repeat_min_thinking_tokens)
        self._ngram_tail.append(token)
        self._ngram_pieces.append(piece)
        repeated = False
        if len(self._ngram_tail) == cfg.repeat_ngram:
            gram = tuple(self._ngram_tail)
            self._ngrams.append(gram)
            self._ngram_counts[gram] += 1
            capacity = cfg.repeat_window - cfg.repeat_ngram + 1
            if len(self._ngrams) > capacity:
                expired = self._ngrams.popleft()
                self._ngram_counts[expired] -= 1
                if not self._ngram_counts[expired]:
                    del self._ngram_counts[expired]
            # Accumulate the exact rolling history before the common gate.
            repeated = (self._ngram_counts[gram] >= cfg.repeat_count
                        and armed and not self._arithmetic(self._ngram_pieces))
        eligible = not self._arithmetic((piece,))
        self._variety_tokens.append((token, eligible))
        if eligible:
            self._variety_counts[token] += 1
            self._variety_eligible += 1
        if len(self._variety_tokens) > cfg.variety_window:
            expired, was_eligible = self._variety_tokens.popleft()
            if was_eligible:
                self._variety_eligible -= 1
                self._variety_counts[expired] -= 1
                if not self._variety_counts[expired]:
                    del self._variety_counts[expired]
        # Budget has priority if several thresholds fire on the same token.
        if not answer and self.thinking_count >= self.effective_budget:
            return "budget"
        if repeated:
            return "repeat_ngram"
        if len(self._variety_tokens) != cfg.variety_window or not self._variety_eligible:
            return None
        if armed and len(self._variety_counts) / self._variety_eligible < cfg.variety_fraction:
            return "low_variety"
        if (cfg.collapse_max_distinct
                and self._variety_eligible >= cfg.collapse_min_eligible
                and len(self._variety_counts) <= cfg.collapse_max_distinct):
            return "collapse"
        return None

    def observe(self, token_id: int, piece: str = "") -> None:
        """Record one actually emitted token, including delimiters and answers."""
        _integer("token_id", token_id)
        if not isinstance(piece, str):
            raise TypeError("piece must be a string")
        expected = bool(self.forced) and self.forced[0] == token_id
        self.total_emitted += 1
        if expected:
            self.forced.popleft()
        if self.closed:
            self.answer_count += 1
            if self.landed and self.answer_trigger is None:
                # Prefix counts toward the cap but never seeds either detector.
                # Without answer_guard only a collapse can end a landed answer.
                reason = None if expected else self._detect(token_id, piece, answer=True)
                if (self.config.answer_guard
                        and self.answer_count >= self.config.answer_max_tokens):
                    reason = "answer_cap"
                if reason is not None:
                    self.answer_trigger = reason
                    self.answer_trigger_count = self.answer_count
                    self._log_event("answer_termination")
            return  # Never enter a new think block after any actual close.
        if token_id == self.close_id:
            if not self.closed:
                self.landed = bool(self.thinking and expected)
                self.thinking = False
                self.closed = True
                if self.landed:
                    # The already-queued answer header survives the forced close.
                    self.answer_count = 0
                    self._reset_detectors()
                    self._log_landing()
                else:
                    self.forced.clear()
            return
        if token_id == self.open_id:
            if not self.thinking:
                self.thinking = True
                self._begin_thinking()
            return
        if not self.thinking:
            return
        self.thinking_count += 1
        if expected:
            self._forced_thinking += 1
            return  # Forced prose cannot feed either loop detector or grace.
        self._sampled_thinking += 1
        if not self.config.enabled:
            return
        boundary = self._sentence_boundary(piece)
        if self.trigger is not None:
            if not self.forced:
                self._grace_used += 1
                if boundary or self._grace_used >= self.config.grace_tokens:
                    self._queue_landing()
            return
        reason = self._detect(token_id, piece)
        if reason is not None:
            if reason == "collapse":
                # Ban the looping word tokens through the landing and the start
                # of the answer, so neither can simply continue the loop.
                self._banned = frozenset(self._variety_counts)
                self._ban_until = self.total_emitted + self.config.collapse_ban_tokens
            self._set_trigger(reason)
            if boundary or self.config.grace_tokens == 0:
                self._queue_landing()

    def __call__(self, token_id: int, piece: str = "") -> None:
        self.observe(token_id, piece)

    @property
    def mask_open(self) -> bool:
        return self.closed

    @property
    def mask_close(self) -> bool:
        return self.closed

    @property
    def banned_ids(self) -> frozenset[int]:
        """Token IDs the sampler must mask now; empty once the ban has lapsed."""
        return self._banned if self.total_emitted < self._ban_until else frozenset()

    @property
    def should_stop(self) -> bool:
        return self.answer_trigger is not None

    @property
    def termination_reason(self) -> str | None:
        if self.answer_trigger is None:
            return None
        return ("answer_cap" if self.answer_trigger == "answer_cap"
                else "answer_" + self.answer_trigger)

    @property
    def serial_required(self) -> bool:
        return (self.trigger is not None and not self.closed) or bool(self.forced)

    def simulation(self) -> ThinkingLanding:
        """Independent speculative state, without copying or invoking the logger."""
        clone = deepcopy(self, {id(self.logger): None} if self.logger is not None else {})
        clone.logger = None
        return clone

    def close_bias(self) -> float:
        """Additive close-logit bias: zero through the ramp's start fraction."""
        if not self.config.enabled or not self.thinking or self.closed:
            return 0.0
        if self.effective_budget == 0:
            return float(self.config.bias_max)
        progress = self._sampled_thinking / self.effective_budget
        start = self.config.bias_start_fraction
        if progress <= start:
            return 0.0
        ramp = min(1.0, (progress - start) / (1.0 - start))
        return float(self.config.bias_max * ramp)

    def metadata(self) -> dict:
        """JSON-serializable state; mask_open is the parent's answer-phase mask."""
        return {
            "enabled": self.config.enabled,
            "budget": self.config.budget,
            "effective_budget": self.effective_budget,
            "output_cap": self.output_cap,
            "answer_reserve": self.config.answer_reserve,
            "answer_max_tokens": self.config.answer_max_tokens,
            "thinking_triggers": self.config.thinking_triggers,
            "answer_guard": self.config.answer_guard,
            "collapse_max_distinct": self.config.collapse_max_distinct,
            "banned_ids": sorted(self.banned_ids),
            "thinking": self.thinking,
            "closed": self.closed,
            "mask_open": self.mask_open,
            "mask_close": self.mask_close,
            "thinking_count": self.thinking_count,
            "sampled_thinking_tokens": self._sampled_thinking,
            "forced_thinking_tokens": self._forced_thinking,
            "total_emitted": self.total_emitted,
            "trigger": self.trigger,
            "trigger_thinking_count": self._trigger_count,
            "grace_tokens_used": self._grace_used,
            "forced_remaining": len(self.forced),
            "landed": self.landed,
            "answer_count": self.answer_count,
            "answer_trigger": self.answer_trigger,
            "answer_trigger_count": self.answer_trigger_count,
            "termination_reason": self.termination_reason,
            "should_stop": self.should_stop,
            "serial_required": self.serial_required,
        }

    def _log_landing(self) -> None:
        self._log_event("thinking_landing")

    def _log_event(self, name: str) -> None:
        if self.logger is None:
            return
        event = self.metadata()
        event["event"] = name
        message = json.dumps(event, sort_keys=True, allow_nan=False)
        if callable(self.logger):
            self.logger(message)
        else:
            self.logger.info(message)
