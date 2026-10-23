"""Sticky reasoning notes: an effort switch mid-conversation leaves every earlier prompt token unchanged.

A template that writes the effort into its first system block (Qwen3.8 Flash, GLM-5.3) re-renders the whole
conversation when the effort changes. A client (the router) that tracks where the level changed sends, in
``chat_template_kwargs``:

- ``reasoning_prefix``: the level the conversation's beginning was rendered with ("off" or a template effort name);
- ``reasoning_changes``: ``[[index, level], ...]``, ``index`` into the request's ``messages`` as the client sent
  them (0-based); the level's note renders right before ``messages[index]``, ``index == len(messages)`` right before
  the generation prompt.

``enable_thinking``/``reasoning_effort`` keep meaning "this reply". A template that supports the notes (an override,
``TF_CHAT_TEMPLATE``) reads:

- ``reasoning_prefix`` (kwarg): as sent, when valid;
- ``message.reasoning_change`` (a key on the message the note precedes): the level;
- ``reasoning_tail`` (kwarg): the level of a note before the generation prompt.

Index mapping. ``normalize_messages`` merges the leading system/developer messages into one system message and only
renames later instructions, so every other message keeps its place and its keys. The note is therefore attached as
a key on the client's message itself, before normalization, and travels with it. Positions it cannot keep:

- index 0: a note before everything is the beginning's level, so it replaces ``reasoning_prefix``;
- an index inside the leading instruction block (1 .. n-1 of n leading system/developer messages, which merge into
  one): the note moves to the first message after the block (the tail when there is none);
- several changes landing on one message: the last one (the highest index, then the later entry) wins.

Bad input (a level the template does not name, an index out of range or not an int, a malformed entry) is dropped,
never refused. Absent both kwargs nothing changes: the render is the stock template's, byte for byte (a client's own
``reasoning_change`` keys are removed either way, so only this module sets them)."""

from __future__ import annotations

from typing import Any

PREFIX = "reasoning_prefix"
CHANGES = "reasoning_changes"
CHANGE = "reasoning_change"        # message key a template reads
TAIL = "reasoning_tail"            # kwarg: the note before the generation prompt
OFF = "off"
KWARGS = (PREFIX, CHANGES)
_ANY_EFFORT = frozenset({"minimal", "low", "medium", "high", "xhigh", "max"})


def allowed_levels(levels: frozenset[str] | None) -> frozenset[str]:
    """"off" plus the template's effort names (any effort name when the template names none)."""

    return frozenset(levels or _ANY_EFFORT) | {OFF}


def requested(kwargs: Any) -> bool:
    return isinstance(kwargs, dict) and any(k in kwargs for k in KWARGS)


def strip(messages: Any) -> Any:
    """``messages`` without client-set ``reasoning_change`` keys (the same list when none has one)."""

    if not isinstance(messages, list) or not any(isinstance(m, dict) and CHANGE in m for m in messages):
        return messages
    return [{k: v for k, v in m.items() if k != CHANGE} if isinstance(m, dict) else m for m in messages]


def sticky_reasoning(messages: Any, kwargs: Any, levels: frozenset[str] | None = None
                     ) -> tuple[Any, dict[str, Any]]:
    """(messages with ``reasoning_change`` keys, template kwargs ``reasoning_prefix``/``reasoning_tail``) for
    ``kwargs`` (a request's ``chat_template_kwargs``) over the client's ``messages``; see the module doc."""

    messages = strip(messages)
    if not requested(kwargs) or not isinstance(messages, list):
        return messages, {}
    allowed = allowed_levels(levels)
    extra: dict[str, Any] = {}
    prefix = kwargs.get(PREFIX)
    if isinstance(prefix, str) and prefix in allowed:
        extra[PREFIX] = prefix
    n = len(messages)
    lead = 0
    while lead < n and isinstance(messages[lead], dict) and messages[lead].get("role") in ("system", "developer"):
        lead += 1
    valid = []
    changes = kwargs.get(CHANGES)
    for item in changes if isinstance(changes, list) else ():
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        index, level = item
        if type(index) is not int or not 0 <= index <= n or not isinstance(level, str) or level not in allowed:
            continue
        valid.append((index, level))
    marks: dict[int, str] = {}
    for index, level in sorted(valid, key=lambda c: c[0]):       # stable: a later entry at one index wins
        if index == 0:
            extra[PREFIX] = level
            continue
        marks[lead if index < lead else index] = level
    if n in marks:
        extra[TAIL] = marks.pop(n)
    if marks:
        messages = [{**m, CHANGE: marks[i]} if i in marks and isinstance(m, dict) else m
                    for i, m in enumerate(messages)]
    return messages, extra


def active(messages: Any, kwargs: Any) -> bool:
    """Whether a render carries notes (a template kwarg or an annotated message)."""

    if isinstance(kwargs, dict) and (PREFIX in kwargs or TAIL in kwargs):
        return True
    return isinstance(messages, list) and any(isinstance(m, dict) and CHANGE in m for m in messages)
