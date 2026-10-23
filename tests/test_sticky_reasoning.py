"""Sticky reasoning notes (``server.sticky_reasoning``): the router's ``reasoning_prefix``/``reasoning_changes`` reach
every render path as a ``reasoning_change`` key on the message the note precedes (client indices mapped past
``normalize_messages``) plus ``reasoning_prefix``/``reasoning_tail`` template kwargs; bad input is dropped; absent
kwargs change nothing. Also ``TF_CHAT_TEMPLATE``, the template override."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("jinja2")

from tensorfold.server.messages import normalize_messages
from tensorfold.server.sticky_reasoning import CHANGE, PREFIX, TAIL, active, sticky_reasoning, strip

LEVELS = frozenset({"low", "medium", "xhigh"})
S = {"role": "system", "content": "s"}
D = {"role": "developer", "content": "d"}
U = {"role": "user", "content": "u"}
A = {"role": "assistant", "content": "a"}
T = {"role": "tool", "tool_call_id": "c", "content": "t"}
# a template that shows where every note lands
NOTES = ("{% if reasoning_prefix is defined %}P={{ reasoning_prefix }};{% endif %}"
         "{% for m in messages %}{% if m.reasoning_change is defined %}[{{ m.reasoning_change }}]{% endif %}"
         "{{ m.role }}:{{ m.content }};{% endfor %}"
         "{% if reasoning_tail is defined %}[{{ reasoning_tail }}]{% endif %}"
         "{% if add_generation_prompt %}assistant:{% endif %}{# 'low' 'medium' 'xhigh' #}")


def notes(messages):
    return [m.get(CHANGE) for m in messages]


def test_absent_kwargs_change_nothing():
    messages = [S, U, A]
    for kwargs in (None, {}, {"enable_thinking": True}, "x"):
        out, extra = sticky_reasoning(messages, kwargs, LEVELS)
        assert out is messages and extra == {}


def test_client_set_change_keys_are_removed():
    tagged = [S, {**U, CHANGE: "low"}]
    out, extra = sticky_reasoning(tagged, None, LEVELS)
    assert notes(out) == [None, None] and extra == {} and CHANGE in tagged[1]      # the client's list untouched
    assert strip([S, U])[1] is U


def test_changes_land_on_the_client_message_and_the_tail():
    messages = [S, U, A, T, U]
    out, extra = sticky_reasoning(messages, {PREFIX: "low", "reasoning_changes": [[3, "xhigh"], [5, "medium"]]}, LEVELS)
    assert notes(out) == [None, None, None, "xhigh", None]
    assert extra == {PREFIX: "low", TAIL: "medium"}
    assert notes(messages) == [None] * 5                                    # copies, not the client's dicts


def test_index_zero_is_the_prefix_and_leading_instructions_move_past_the_merge():
    messages = [S, D, D, U, A]
    out, extra = sticky_reasoning(messages, {PREFIX: "low", "reasoning_changes": [[0, "medium"], [2, "xhigh"]]},
                                  LEVELS)
    assert extra == {PREFIX: "medium"} and notes(out) == [None, None, None, "xhigh", None]
    merged = normalize_messages(out)                       # the leading three merge: the note stays on the user
    assert [m["role"] for m in merged] == ["system", "user", "assistant"] and notes(merged) == [None, "xhigh", None]
    only = [S, D]
    out, extra = sticky_reasoning(only, {"reasoning_changes": [[1, "low"]]}, LEVELS)
    assert extra == {TAIL: "low"} and notes(out) == [None, None]


def test_late_instructions_keep_their_note_through_normalization():
    messages = [S, U, A, {"role": "system", "content": "late"}, U]
    out, _ = sticky_reasoning(messages, {"reasoning_changes": [[3, "low"], [4, "xhigh"]]}, LEVELS)
    for late in ("system", "user"):
        normal = normalize_messages(out, late_system=late)
        assert notes(normal) == [None, None, None, "low", "xhigh"] and normal[3]["role"] == late


def test_later_change_at_one_place_wins_and_order_does_not_matter():
    out, extra = sticky_reasoning([S, U, A, U], {"reasoning_changes": [[3, "low"], [1, "medium"], [3, "xhigh"]]},
                                  LEVELS)
    assert notes(out) == [None, "medium", None, "xhigh"] and extra == {}


@pytest.mark.parametrize("bad", [[9, "low"], [-1, "low"], ["1", "low"], [True, "low"], [1.0, "low"], [1, "high"],
                                 [1, None], [1], [1, "low", 2], "x", None, {"1": "low"}])
def test_bad_changes_are_dropped(bad):
    out, extra = sticky_reasoning([S, U, A], {PREFIX: "low", "reasoning_changes": [bad, [2, "xhigh"]]}, LEVELS)
    assert notes(out) == [None, None, "xhigh"] and extra == {PREFIX: "low"}


@pytest.mark.parametrize("changes", ["x", 3, {"1": "low"}, None])
@pytest.mark.parametrize("prefix", ["high", "", 3, None, "OFF"])
def test_bad_prefix_and_changes_are_dropped(prefix, changes):
    out, extra = sticky_reasoning([S, U], {PREFIX: prefix, "reasoning_changes": changes}, LEVELS)
    assert extra == {} and notes(out) == [None, None]


def test_off_is_always_a_level_and_any_effort_without_template_names():
    assert sticky_reasoning([U], {PREFIX: "off"}, LEVELS)[1] == {PREFIX: "off"}
    assert sticky_reasoning([U], {PREFIX: "high"}, frozenset())[1] == {PREFIX: "high"}


def test_active():
    assert not active([U], {}) and active([U], {PREFIX: "low"}) and active([U], {TAIL: "low"})
    assert active([{**U, CHANGE: "low"}], None)


# -- render paths ---------------------------------------------------------------------------------


class JinjaTokenizer:
    """A tokenizer whose chat template renders as text (one token a character), recording each call's kwargs."""

    def __init__(self, template=NOTES):
        self.chat_template = template
        self.calls = []

    def apply_chat_template(self, messages, tokenize=True, **kwargs):
        import jinja2

        self.calls.append(kwargs)
        text = jinja2.Environment().from_string(self.chat_template).render(messages=messages, **kwargs)
        return [ord(c) for c in text] if tokenize else text

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)


def text(ids):
    return "".join(chr(i) for i in ids)


def test_render_prompt_ids_passes_the_template_kwargs():
    from tensorfold.server.text import render_prompt_ids

    tok = JinjaTokenizer()
    messages, extra = sticky_reasoning([S, U, A, U], {PREFIX: "low", "reasoning_changes": [[3, "xhigh"], [4, "low"]]},
                                       LEVELS)
    ids = render_prompt_ids(tok, messages, enable_thinking=True, late_system="system", template_kwargs=extra)
    assert text(ids) == "P=low;system:s;user:u;assistant:a;[xhigh]user:u;[low]assistant:"
    plain = render_prompt_ids(tok, [S, U], enable_thinking=True, late_system="system")
    assert text(plain) == "system:s;user:u;assistant:" and PREFIX not in tok.calls[-1]


def test_the_mac_app_renders_the_request_s_notes_and_its_probe_keeps_the_prefix():
    from tensorfold.server.prompt_blocks import _REQUEST, PromptBlocks

    tok = JinjaTokenizer()
    app = SimpleNamespace(tokenizer=tok, tokenizer_lock=threading.Lock(), enable_thinking=True,
                          late_system="system", effort_for=lambda effort: effort)
    messages, extra = sticky_reasoning([S, U, A, U], {PREFIX: "low", "reasoning_changes": [[1, "medium"], [4, "xhigh"]]},
                                       LEVELS)
    _REQUEST.sampling = {"template_kwargs": extra}
    try:
        prompt, history = PromptBlocks.render(app, messages, None)
        assert text(prompt) == "P=low;system:s;[medium]user:u;assistant:a;user:u;[xhigh]assistant:"
        assert text(prompt[:history]) == "P=low;system:s;[medium]user:u;assistant:a;user:u;[xhigh]"
        PromptBlocks.system_prefix_len(app, messages, None, prompt)
        assert tok.calls[-1].get(PREFIX) == "low" and TAIL not in tok.calls[-1]      # the probe has no tail
    finally:
        _REQUEST.sampling = None


def test_the_mac_http_route_annotates_before_normalizing():
    from tests.test_server_openai_compat import FakeApp, post_json, serve_fake

    class SamplingApp(FakeApp):
        accepts_sampling = True
        effort_levels = LEVELS

        def chat(self, messages, *, sampling=None, **kwargs):
            self.sampling = sampling
            return super().chat(messages, **kwargs)

    app = SamplingApp()
    server = serve_fake(app)
    try:
        body = {"model": "fake-model", "messages": [S, D, U, A, {"role": "developer", "content": "late"}, U],
                "chat_template_kwargs": {PREFIX: "low", "reasoning_changes": [[1, "medium"], [4, "low"], [6, "xhigh"]]}}
        status, _ = post_json(server, "/v1/chat/completions", body)
    finally:
        server.shutdown()
    assert status == 200
    assert [m["role"] for m in app.messages] == ["system", "user", "assistant", "system", "user"]
    assert notes(app.messages) == [None, "medium", None, "low", None]
    assert app.sampling["template_kwargs"] == {PREFIX: "low", TAIL: "xhigh"}


def test_the_mac_tokenize_route_renders_the_notes():
    from tensorfold.server.token_routes import tokenize

    tok = JinjaTokenizer()
    app = SimpleNamespace(tokenizer=tok, tokenizer_lock=threading.Lock(), enable_thinking=True, effort_levels=LEVELS,
                          late_system="system", effort_for=lambda effort: effort, context_window=0)
    reply = tokenize(app, {"messages": [S, U, A, U],
                           "chat_template_kwargs": {PREFIX: "off", "reasoning_changes": [[3, "low"], [4, "xhigh"]]}})
    assert text(reply["tokens"]) == "P=off;system:s;user:u;assistant:a;[low]user:u;[xhigh]assistant:"


def test_the_cuda_server_maps_the_notes_and_passes_no_raw_kwargs(tmp_path):
    from tensorfold.cuda import server
    from tests.test_cuda_tool_choice import Tokens

    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": NOTES + "{{ reasoning_changes }}"}))
    app = server.App.__new__(server.App)
    app.engine, app.served, app.tok = None, "fake-cuda", Tokens()
    app.template = server.ChatTemplate(tmp_path)
    app.default_thinking, app.reasoning_effort, app.thinking_budget = True, None, 0
    app.sampling, app.max_tokens = {"temperature": 0.0}, 8
    app.native_context_window = app.context_window = 0
    body = {"messages": [S, D, U, A, U],
            "chat_template_kwargs": {PREFIX: "low", "reasoning_changes": [[1, "medium"], [4, "bad"], [5, "xhigh"]]}}
    got = Tokens().decode(app._prepare(body, True).prompt)
    assert got == "P=low;system:s\n\nd;[medium]user:u;assistant:a;user:u;[xhigh]assistant:"
    plain = Tokens().decode(app._prepare({"messages": [S, U]}, True).prompt)
    assert plain == "system:s;user:u;assistant:"


def test_glm_thinking_off_keeps_the_effort_lines_of_a_sticky_render():
    from tensorfold.families.glm5_next.prompts import thinking_off

    text = "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>u<|assistant|><think>"
    assert thinking_off(text) == "[gMASK]<sop><|user|>u<|assistant|><think></think>"
    assert thinking_off(text, sticky=True) == text + "</think>"


# -- TF_CHAT_TEMPLATE -----------------------------------------------------------------------------


def test_the_override_replaces_the_checkpoint_template(tmp_path, monkeypatch):
    from tensorfold.cuda.chat_template import ChatTemplate
    from tensorfold.server.thinking_notes import template_thinks

    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages[0].content }}"}))
    stock = ChatTemplate(tmp_path)
    assert stock.late_system == "user" and stock.efforts == frozenset() and not template_thinks(tmp_path)
    override = tmp_path / "override.jinja"
    override.write_text("{% for m in messages %}{{ m.content }}{% endfor %}{{ enable_thinking }}{# 'low' 'max' #}")
    monkeypatch.setenv("TF_CHAT_TEMPLATE", str(override))
    over = ChatTemplate(tmp_path)
    assert over.late_system == "system" and over.efforts == frozenset({"low", "max"}) and template_thinks(tmp_path)
    assert over.render([U], tools=None, enable_thinking=True) == "uTrue"
    monkeypatch.setenv("TF_CHAT_TEMPLATE", str(tmp_path / "missing.jinja"))
    with pytest.raises(FileNotFoundError):
        ChatTemplate(tmp_path)
    monkeypatch.setenv("TF_CHAT_TEMPLATE", "")
    assert ChatTemplate(tmp_path).late_system == "user"


def test_the_override_reaches_wrapped_mac_tokenizers(tmp_path, monkeypatch):
    from tensorfold.families.glm5_next.prompts import GlmTokenizer
    from tensorfold.families.tokenizer import with_template

    inner = SimpleNamespace(chat_template="stock", _tokenizer=SimpleNamespace(chat_template="stock"))
    wrapped = GlmTokenizer(inner)
    assert with_template(wrapped) is wrapped and inner.chat_template == "stock"           # unset: untouched
    override = tmp_path / "t.jinja"
    override.write_text("new")
    monkeypatch.setenv("TF_CHAT_TEMPLATE", str(override))
    with_template(wrapped)
    assert inner.chat_template == inner._tokenizer.chat_template == wrapped.chat_template == "new"
