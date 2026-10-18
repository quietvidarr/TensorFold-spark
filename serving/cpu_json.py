"""CPU integration with checkpoint tokenizer, real xgrammar and runtime code.

Only numerical forward/commit/cache storage are doubles. Actual HTTP prepare/run,
owned decode loops, Engine sample/verify/follow, prefill, absorb, draft, sampler,
and grammar execute. This is NOT a GPU cache/numerical or model-quality proof.
"""
import ast
import contextlib
import dataclasses
import hashlib
import http.client
from http.server import ThreadingHTTPServer
import importlib.metadata
import importlib.util
import json
import os
import socket
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import torch
from tokenizers import Tokenizer
from tensorfold.engine import grammar
from tensorfold.engine.exact_sampling import Sampling, MARGIN, choose_rows
from tensorfold.cuda.sampling import nucleus_rows, one_rank, comm_gather
from tensorfold.server.errors import RequestError
from tensorfold.server.cancellation import RequestCancelled

COPY = Path(__file__).resolve().parents[1]
MODEL = Path(os.environ.get('TF_JSON_CPU_MODEL', '/model'))
TOK = Tokenizer.from_file(str(MODEL / 'tokenizer.json'))
VOCAB = grammar.vocab_size(MODEL)
EOS = (154820, 154827, 154829)
OPEN = TOK.token_to_id('<think>')
CLOSE = TOK.token_to_id('</think>')
GRAMMARS = grammar.Grammars.for_model(MODEL, VOCAB, EOS)
SPEC = grammar.Spec('json')
COMPILED = GRAMMARS.compile(SPEC)

def enc(s):
    return TOK.encode(s, add_special_tokens=False).ids

def constraint(thinking=False):
    return GRAMMARS.constraint(COMPILED, think_end=CLOSE if thinking else None, spec=SPEC)

path = COPY / 'serving/server.py'
spec = importlib.util.spec_from_file_location('v54_json_server', path)
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)
source = COPY / 'src/tensorfold/families/glm5_next/cuda/decode.py'
tree = ast.parse(source.read_text())
engine = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Engine')
nodes = [n for n in engine.body if isinstance(n, ast.FunctionDef) and n.name in ('sample', 'verify_window', 'follow')]
nodes += [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Snapshot']
nodes += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in
          ('sample_rows', '_probability', 'prefill', '_absorb_rows', 'absorb', 'draft', 'take_snapshot', 'restore')]
unit = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + nodes, type_ignores=[])
ast.fix_missing_locations(unit)
NSPACE = dict(torch=torch, np=np, dataclass=dataclasses.dataclass, Sampling=Sampling, MARGIN=MARGIN, choose_rows=choose_rows,
              nucleus_rows=nucleus_rows, one_rank=one_rank, comm_gather=comm_gather)
exec(compile(unit, str(source), 'exec'), NSPACE)
torch.set_num_threads(1)

class CPUModel:
    """Synthetic logits, exact positional hidden ledger, real token-control code."""
    sample = NSPACE['sample']
    verify_window = NSPACE['verify_window']

    def __init__(self, sequence, adversarial=False, bad_draft=False):
        self.sequence = sequence
        self.adversarial, self.bad_draft = adversarial, bad_draft
        self.constraint = self.window = self.thinking_landing = None
        self.w = NS(cfg=NS(eos=EOS, mtp_layers=1), mtp=object(), vocab_offset=0, comm=None)
        # rec is a recurrent ledger: every committed token moves it, so a resumed prompt must restore it exactly
        self.st = NS(pos=0, capacity=89344, mtp_len=0, mtp_drafted=0, cur=[0],
                     rec=torch.zeros(2, 1, dtype=torch.float64), conv=torch.zeros(1, dtype=torch.float64))
        self.st.set_mtp_len = lambda n: setattr(self.st, 'mtp_len', n)
        self.st.set_pos = self.set_pos
        self.buf, self.pbuf, self.mbuf = NS(prefill=False), NS(prefill=True), NS(rows=2)
        self.graphs, self.replays = None, {}
        self.scratch = torch.empty(2, 1)
        self.history, self.commits, self.pairs, self.draft_counts = [], [], [], []
        self.last_hidden = torch.zeros(1, 1)
        self.fence = enc('```')[0]
        self.context = []
        self.prefills = 0
        self.filled = []                          # prompt rows each prefill computed
        self.api = NS(prefill=self.prefill, draft=self.draft, commit=self.commit, take_snapshot=NSPACE['take_snapshot'])
        cuda = NS(synchronize=lambda: None, max_memory_reserved=lambda: 0)
        self.runtime = (self, TOK, NS(inference_mode=torch.inference_mode, cuda=cuda),
                        self.api, Sampling, self.scratch, None, NS(guard=lambda phase: None), {})
        self.app = server.make_app(self.runtime, MODEL)
        self.app.grammars = GRAMMARS

    def reset(self):
        self.st.pos = self.st.mtp_len = self.st.mtp_drafted = 0
        self.st.rec.zero_()
        self.st.conv.zero_()
        self.history = []
        self.context = []

    def set_pos(self, pos):                        # restore(): the committed context is the kept prefix again
        self.st.pos = pos
        del self.context[pos:]

    def wanted(self, index):
        return self.sequence[index] if index < len(self.sequence) else EOS[0]

    def logits(self, token, adversarial=False):
        y = torch.full((1, VOCAB), -100.0)
        y[0, token] = 20
        if adversarial:
            # Fences and premature EOS must lose to a lower-logit legal token.
            y[0, self.fence] = 100
            if token not in EOS:
                y[0, EOS[0]] = 90
        return y

    def follow(self, tokens):
        NSPACE['follow'](self, tokens)
        self.history.extend(tokens)

    def prefill(self, e, prompt, sampling, **kwargs):
        self.prefills += 1
        self.prefill_rows = len(prompt) + 1
        self.history = []                          # this request's reply: a resumed prompt has no engine reset to clear it
        self.filled.append(0)
        def stage(w, st, b, tokens):
            b.tokens = list(tokens)
            self.filled[-1] += len(tokens)
            return len(tokens)
        def compute(w, st, b, rows, **kw):
            b.fnormed = torch.arange(st.pos, st.pos + rows, dtype=torch.float32).view(-1, 1)
            return self.logits(self.wanted(0), self.adversarial and self.constraint.active)
        def mtp_forward(w, st, b, tokens, hidden):
            for i, t in enumerate(tokens):
                assert int(hidden[i, 0]) == st.mtp_len + i
        NSPACE.update(stage=stage, compute=compute, chunks_for=lambda *a: 1,
                      commit=self.commit, mtp_forward=mtp_forward,
                      prof=NS(active=False, timed=lambda *a: contextlib.nullcontext(), report=lambda *a: None))
        return NSPACE['prefill'](e, prompt, sampling, **kwargs)

    def forward(self, tokens):
        self.buf.tokens = list(tokens)
        self.hiddens = torch.arange(self.st.pos, self.st.pos + len(tokens), dtype=torch.float32).view(-1, 1)
        self.graph_output = torch.cat([self.logits(self.wanted(len(self.history) + i),
                      self.adversarial and self.constraint is not None and self.constraint.active)
                      for i in range(len(tokens))])
        self.graph_original = self.graph_output.clone()
        return self.graph_output

    def commit(self, w, st, b, rows, keep):
        for token in b.tokens[:keep]:
            st.rec[st.cur[0]] = (st.rec[st.cur[0]] * 31 + token) % 1000003
        self.context.extend(b.tokens[:keep])
        st.pos += keep
        if not b.prefill:
            self.commits.append((rows, keep))
            assert torch.equal(self.graph_output, self.graph_original), 'graph-owned logits mutated'

    def mtp(self, tokens, hidden):
        for i, token in enumerate(tokens):
            position = self.st.mtp_len + i
            assert int(hidden[i, 0]) == position, ('MTP hidden alignment', position, hidden.tolist())
            self.pairs.append((position, token))
        return self.logits(self.fence if self.bad_draft else self.wanted(len(self.history)))

    def main_hidden(self, rows):
        return self.hiddens[rows]

    def draft(self, *args):
        self.draft_counts.append(args[4])
        return NSPACE['draft'](*args)

    def run(self, **kw):
        body = dict(model='glm53', messages=[dict(role='user', content='Return an object.')],
                    max_tokens=4096, response_format={'type': 'json_object'}, return_token_ids=True)
        body.update(kw)
        self.body = body
        self.prepared = self.app.prepare(body, True)
        if self.prepared.landing_state is not None:
            self.prepared.landing_state.logger = None
        self.deltas = []
        result = self.app.run(body, True, lambda d: self.deltas.append(d) or True, prepared=self.prepared)
        assert self.constraint is None and self.window is None and self.thinking_landing is None
        return result


class GrammarContracts(unittest.TestCase):
    def allowed(self, c, token):
        y = torch.zeros(1, VOCAB)
        c.mask(y)
        return bool(torch.isfinite(y[0, token]))

    def test_fences_commentary_arrays_eos_rejected_before_object(self):
        for text in ('```json\n', 'Here is JSON:', '[1,2]', 'true', 'null'):
            with self.subTest(text=text):
                self.assertFalse(self.allowed(constraint(), enc(text)[0]))
        for eos in EOS:
            self.assertFalse(self.allowed(constraint(), eos))

    def test_real_json_unicode_escape_nested_and_eos(self):
        text = '{"name":"A\\\"B","城市":"北京","x":[1,true,null,{"ok":false}]}'
        c = constraint()
        for token in enc(text):
            self.assertTrue(self.allowed(c, token), (token, TOK.decode([token])))
            c.advance([token])
        self.assertFalse(c.finished)
        for eos in EOS:
            self.assertTrue(self.allowed(c, eos))
        self.assertFalse(self.allowed(c, enc('```')[0]))
        c.advance([EOS[0]])
        self.assertTrue(c.finished)
        self.assertIsInstance(json.loads(text), dict)

    def test_reasoning_boundary_and_speculative_rollback(self):
        c = constraint(True)
        fence = enc('```')[0]
        c.advance(enc('unconstrained prose ``` [ ]'))
        self.assertFalse(c.active)
        before = c.window([0], [-1])
        self.assertEqual(before.rows, [])
        window = c.window([0, CLOSE, fence], [-1, 0, 1])
        self.assertEqual(window.tokens, [0, CLOSE])
        self.assertEqual(window.rows, [1])
        self.assertFalse(c.active)  # speculation did not advance committed phase
        c.advance([CLOSE])
        start = c.window([0], [-1]).bits.copy()
        good = enc('{"k":1}')
        window = c.window([0] + good + [fence], list(range(-1, len(good) + 1)))
        self.assertEqual(window.tokens, [0] + good)
        self.assertTrue(np.array_equal(start, c.window([0], [-1]).bits))
        c.advance(good)
        c.advance([EOS[0]])
        self.assertTrue(c.finished)

    def test_bad_committed_token_fails_closed(self):
        with self.assertRaises(grammar.GrammarError):
            constraint().advance(enc('```'))


class IntegrationContracts(unittest.TestCase):
    def test_ordinary_chat_keeps_mtp_and_unconstrained_answer(self):
        seq = enc('Think carefully.') + [CLOSE] + enc('```text\nordinary answer\n```') + [EOS[0]]
        f = CPUModel(seq)
        result = f.run(response_format=None)
        self.assertEqual(result['stats']['token_ids'], seq)
        self.assertTrue(result['content'].startswith('```text'))
        self.assertNotIn('structured_output', result['stats'])
        self.assertGreater(result['stats']['accepted'], 0)

    def test_cancelled_callback_cleans_matcher(self):
        f = CPUModel(enc('Thinking.') + [CLOSE] + enc('{}') + [EOS[0]])
        body = dict(messages=[dict(role='user', content='x')], max_tokens=4096,
                    response_format={'type': 'json_object'})
        with self.assertRaises(RequestCancelled):
            f.app.run(body, True, lambda delta: False)
        self.assertIsNone(f.constraint)
        self.assertIsNone(f.window)
        self.assertIsNone(f.thinking_landing)

    def test_real_http_json_sse_and_400_admission(self):
        from tensorfold.cuda.http import make_handler
        seq = [CLOSE] + enc('{"wire":"ok"}') + [EOS[0]]
        f = CPUModel(seq)
        httpd = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(f.app))
        worker = threading.Thread(target=httpd.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
        worker.start()
        def post(body):
            conn = http.client.HTTPConnection('127.0.0.1', httpd.server_port, timeout=10)
            try:
                conn.request('POST', '/v1/chat/completions', json.dumps(body), {'Content-Type': 'application/json'})
                response = conn.getresponse()
                return response.status, response.read().decode()
            finally:
                conn.close()
        body = dict(model='glm53', messages=[dict(role='user', content='object')],
                    max_tokens=4096, response_format={'type': 'json_object'}, return_token_ids=True)
        try:
            status, text = post(body)
            self.assertEqual(status, 200)
            payload = json.loads(text)
            self.assertEqual(json.loads(payload['choices'][0]['message']['content']), {'wire': 'ok'})
            self.assertEqual(payload['choices'][0]['finish_reason'], 'stop')
            self.assertEqual(payload['tensorfold']['token_ids'], seq)
            status, text = post({**body, 'stream': True})
            self.assertEqual(status, 200)
            self.assertTrue(text.rstrip().endswith('data: [DONE]'))
            events = [json.loads(line[6:]) for line in text.splitlines() if line.startswith('data: {')]
            content = ''.join(e['choices'][0]['delta'].get('content', '') for e in events if e.get('choices'))
            self.assertEqual(json.loads(content), {'wire': 'ok'})
            self.assertEqual(events[-1]['usage']['completion_tokens'], len(seq))
            self.assertTrue(events[-1]['tensorfold']['structured_output']['grammar_terminated'])
            before = f.prefills
            status, text = post({**body, 'response_format': {'type': 'json_schema', 'json_schema': {'schema': {'type': 'object'}}}})
            self.assertEqual(status, 400)
            self.assertIn('error', json.loads(text))
            self.assertEqual(f.prefills, before)
        finally:
            httpd.shutdown()
            httpd.server_close()
            worker.join()

    def test_real_prepare_preserves_prompt_defaults_and_no_header(self):
        f = CPUModel([])
        body = dict(messages=[dict(role='user', content='An object please.')], max_tokens=4096)
        ordinary = f.app.prepare(body, True)
        shaped = f.app.prepare({**body, 'response_format': {'type': 'json_object'}}, True)
        self.assertEqual(ordinary.prompt, shaped.prompt)  # no JSON prompt injection
        self.assertEqual(shaped.sampling.temperature, 0.3)
        self.assertEqual(shaped.landing_state.answer_prefix_ids, ())
        self.assertEqual(shaped.landing_state.config, ordinary.landing_state.config)
        self.assertEqual(shaped.grammar[0], SPEC)

    def test_http_run_reasoning_then_valid_json_with_mtp(self):
        seq = enc('I will calculate carefully.') + [CLOSE] + enc('{"sum":17,"ok":true}') + [EOS[0]]
        f = CPUModel(seq)
        result = f.run()
        self.assertEqual(result['stats']['token_ids'], seq)
        self.assertEqual(json.loads(result['content']), {'sum': 17, 'ok': True})
        self.assertTrue(result['reasoning'])
        self.assertEqual(result['finish'], 'stop')
        self.assertGreater(result['stats']['accepted'], 0)
        self.assertEqual(f.history, seq)
        self.assertEqual(''.join(d.get('content', '') for d in f.deltas) + result['final'].get('content', ''), result['content'])
        self.assertEqual(f.context, f.prepared.prompt + seq[:-1])
        self.assertTrue(result['stats']['structured_output']['grammar_terminated'])

    def test_zero_budget_forced_close_then_json_and_mtp_resume(self):
        seq = [CLOSE] + enc('{"answer":42}') + [EOS[0]]
        f = CPUModel(seq)
        result = f.run(thinking_budget=0)
        self.assertEqual(result['stats']['token_ids'], seq)
        self.assertEqual(json.loads(result['content']), {'answer': 42})
        self.assertGreater(result['stats']['accepted'], 0)
        self.assertGreater(result['stats']['mtp_resumes'], 0)

    def test_forced_landing_prose_stays_reasoning_and_mtp_resumes(self):
        phrase = enc(server.LANDING_TEXT)
        seq = enc('Wait') + phrase + [CLOSE] + enc('{"verified":false}') + [EOS[0]]
        f = CPUModel(seq)
        result = f.run(thinking_loop={'budget': 1, 'grace_tokens': 0})
        self.assertEqual(result['stats']['token_ids'], seq)
        self.assertIn(server.LANDING_TEXT, result['reasoning'])
        self.assertEqual(json.loads(result['content']), {'verified': False})
        self.assertGreater(result['stats']['mtp_resumes'], 0)
        self.assertIn(0, f.draft_counts)
        self.assertGreater(result['stats']['accepted'], 0)

    def test_active_first_token_mask_blocks_higher_fence_and_eos(self):
        # Every target row offers higher-logit invalid tokens. Use an empty
        # object so a fence is never a legal string-content alternative.
        seq = enc('{}') + [EOS[0]]
        f = CPUModel(seq, adversarial=True, bad_draft=True)
        from tensorfold.engine.thinking_landing import ThinkingLanding
        state = ThinkingLanding(f.app.landing_defaults, OPEN, CLOSE, enc(server.LANDING_TEXT),
                                4096, initial_thinking=False, answer_prefix_ids=())
        out = []
        stats = f.app.engine.generate([1, OPEN, CLOSE], 4096, Sampling(1, .3, 0, .95, .05),
                                     lambda ts: out.extend(ts), landing_state=state, constraint=constraint())
        self.assertEqual(out, seq)
        self.assertEqual(stats['accepted'], 0)
        self.assertGreater(stats['drafted'], 0)
        self.assertTrue(stats['structured_output']['grammar_terminated'])

    def test_serial_and_mtp_match_with_rejected_proposals(self):
        seq = [CLOSE] + enc('{"a":7,"b":9}') + [EOS[0]]
        f = CPUModel(seq, bad_draft=True)
        g = CPUModel(seq, bad_draft=True)
        a = f.run(thinking_budget=0)
        b = g.run(thinking_budget=0, draft=False)
        self.assertEqual(a['stats']['token_ids'], b['stats']['token_ids'])
        self.assertEqual(a['content'], b['content'])
        self.assertEqual(a['stats']['accepted'], 0)
        self.assertEqual(a['stats']['mtp_n'], 1)
        self.assertEqual(b['stats']['mtp_n'], 0)
        self.assertTrue(all(t != f.fence for t in f.history))

    def test_landing_disabled_still_masks_prefill_and_windows(self):
        f = CPUModel([CLOSE] + enc('{}') + [EOS[0]], adversarial=True, bad_draft=True)
        result = f.run(thinking_loop={'enabled': False})
        self.assertEqual(result['content'], '{}')
        self.assertEqual(result['finish'], 'stop')

    def test_caps_return_length_and_never_overshoot(self):
        for count in (1, 2, 3, 4, 5):
            f = CPUModel([CLOSE] + enc('{"value":1234567890123456789}') + [EOS[0]])
            result = f.run(thinking_loop={'enabled': False}, max_tokens=count)
            self.assertEqual(result['completion_tokens'], count)
            self.assertEqual(result['finish'], 'length')
            self.assertFalse(result['stats']['structured_output']['grammar_terminated'])

    def test_protective_answer_cap_not_false_json_success(self):
        seq = [CLOSE] + enc('{"unfinished":1234567890123456789}') + [EOS[0]]
        f = CPUModel(seq)
        result = f.run(thinking_loop={'budget': 0, 'answer_max_tokens': 1, 'answer_guard': True})
        self.assertEqual(result['finish'], 'length')
        self.assertEqual(result['stats']['thinking_loop']['termination_reason'], 'answer_cap')
        self.assertFalse(result['stats']['structured_output']['grammar_terminated'])

    def test_default_landed_answer_is_not_capped(self):
        # v55 defaults: answer_guard off, so a landed answer ends naturally.
        f = CPUModel([CLOSE] + enc('{"unfinished":1234567890123456789}') + [EOS[0]])
        result = f.run(thinking_loop={'budget': 0, 'answer_max_tokens': 1})
        self.assertEqual(result['content'], '{"unfinished":1234567890123456789}')
        self.assertEqual(result['finish'], 'stop')
        self.assertIsNone(result['stats']['thinking_loop']['termination_reason'])
        self.assertTrue(result['stats']['structured_output']['grammar_terminated'])

    def test_background_does_not_restart_matcher(self):
        f = CPUModel([CLOSE] + enc('{"x":1}') + [EOS[0]])
        with patch('tensorfold.cuda.turns.Yield', side_effect=AssertionError('grammar must not yield')):
            result = f.run(priority='background')
        self.assertEqual(f.prefills, 1)
        self.assertEqual(result['finish'], 'stop')

    def test_runtime_exception_restores_all_request_hooks(self):
        f = CPUModel([CLOSE] + enc('{}') + [EOS[0]])
        def fail(*a):
            raise RuntimeError('synthetic forward failure')
        f.forward = fail
        with self.assertRaisesRegex(RuntimeError, 'synthetic forward'):
            f.run()
        self.assertIsNone(f.constraint)
        self.assertIsNone(f.window)
        self.assertIsNone(f.thinking_landing)

    def test_unqualified_combinations_fail_before_generation(self):
        f = CPUModel([])
        base = dict(messages=[dict(role='user', content='x')], response_format={'type': 'json_object'})
        bad = [dict(response_format={'type': 'json_schema', 'json_schema': {'schema': {'type': 'object'}}}),
               dict(response_format={'type': 'json_object', 'unknown': True}),
               dict(tools=[{'type': 'function', 'function': {'name': 'x', 'parameters': {'type': 'object'}}}]),
               dict(tool_choice='auto'), dict(ignore_eos=True), dict(stop=['}']),
               dict(reasoning_effort='none'), dict(chat_template_kwargs={'enable_thinking': False}),
               dict(guided_regex='.*'), dict(guided_choice=['a']), dict(guided_grammar='root ::= "a"'),
               dict(structured_outputs={'json_object': True}),
               dict(messages=[{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'https://invalid/image.png'}}]}])]
        for fields in bad:
            with self.subTest(fields=fields), self.assertRaises(RequestError):
                f.app.prepare({**base, **fields}, True)
        with self.assertRaises(RequestError):
            f.app.prepare({'prompt': 'x', 'response_format': {'type': 'json_object'}}, False)
        self.assertEqual(f.prefills, 0)

    def test_missing_dependency_is_request_error(self):
        real_import = __import__
        def importing(name, *args, **kwargs):
            if name == 'xgrammar':
                raise ImportError('test absent optional dependency')
            return real_import(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=importing), self.assertRaises(RequestError):
            grammar.Grammars.for_model(MODEL, VOCAB, EOS)

    def test_actual_absorb_rolls_back_drafted_cache_ledger(self):
        f = CPUModel([])
        f.st.mtp_len = 7
        f.st.mtp_drafted = 2
        NSPACE['absorb'](f, torch.tensor([[5.], [6.]]), enc('{}')[:1] + [EOS[0]])
        self.assertEqual(f.st.mtp_len, 7)
        self.assertEqual(f.st.mtp_drafted, 0)
        self.assertEqual([p for p, t in f.pairs], [5, 6])


class PrefixResume(unittest.TestCase):
    """v57: a prompt that extends the live one resumes from it and ends in a fresh run's state."""

    SEQ = [CLOSE] + enc('{"n":1}') + [EOS[0]]
    FIRST = [dict(role='user', content='Return an object.')]
    SECOND = FIRST + [dict(role='assistant', content='{"n":1}'), dict(role='user', content='Again, please.')]

    @staticmethod
    def state(f):
        return f.st.pos, f.st.mtp_len - f.st.mtp_drafted, f.st.rec[f.st.cur[0]].tolist(), list(f.context)

    def test_next_turn_resumes_and_ends_in_a_fresh_runs_state(self):
        a = CPUModel(self.SEQ)
        one = a.run(messages=self.FIRST)
        n = one['prompt_tokens']
        two = a.run(messages=self.SECOND)
        b = CPUModel(self.SEQ)
        fresh = b.run(messages=self.SECOND, prefix_cache=False)
        self.assertEqual(one['cached_tokens'], 0)
        self.assertEqual((two['cached_tokens'], two['stats']['cached']), (n, n))
        self.assertEqual(fresh['cached_tokens'], 0)
        self.assertEqual(a.filled, [n, two['prompt_tokens'] - n])      # only the new rows were computed
        self.assertEqual(b.filled, [two['prompt_tokens']])
        self.assertEqual(two['stats']['token_ids'], fresh['stats']['token_ids'])
        self.assertEqual(two['content'], fresh['content'])
        self.assertEqual(self.state(a), self.state(b))

    def test_identical_resend_resumes_from_the_earlier_state(self):
        a = CPUModel(self.SEQ)
        n = a.run(messages=self.FIRST)['prompt_tokens']
        first = a.run(messages=self.SECOND)
        again = a.run(messages=self.SECOND)
        self.assertEqual(again['cached_tokens'], n)      # the newest state equals the prompt: not a strict prefix
        self.assertEqual(again['stats']['token_ids'], first['stats']['token_ids'])
        lone = CPUModel(self.SEQ)
        lone.run(messages=self.FIRST)
        self.assertEqual(lone.run(messages=self.FIRST)['cached_tokens'], 0)   # nothing shorter was kept

    def test_other_conversation_starts_from_a_reset_and_drops_the_old_state(self):
        a = CPUModel(self.SEQ)
        a.run(messages=self.FIRST)
        other = a.run(messages=[dict(role='user', content='Something else entirely.')])
        self.assertEqual(other['cached_tokens'], 0)
        self.assertEqual(a.run(messages=self.SECOND)['cached_tokens'], 0)
        self.assertEqual(len(server.prefix_states(a).snaps), 1)

    def test_request_opt_out_fills_whole_prompt_and_still_keeps_its_state(self):
        a = CPUModel(self.SEQ)
        n = a.run(messages=self.FIRST)['prompt_tokens']
        out = a.run(messages=self.SECOND, prefix_cache=False)
        self.assertEqual(out['cached_tokens'], 0)
        self.assertEqual(a.filled[-1], out['prompt_tokens'])
        third = self.SECOND + [dict(role='assistant', content='{"n":1}'), dict(role='user', content='Once more.')]
        self.assertEqual(a.run(messages=third)['cached_tokens'], out['prompt_tokens'])
        with self.assertRaises(RequestError):
            a.app.prepare(dict(model='glm53', messages=self.FIRST, max_tokens=4096, prefix_cache='yes'), True)
        self.assertGreater(n, 0)

    def test_cancelled_prompt_computes_nothing_and_drops_every_kept_state(self):
        a = CPUModel(self.SEQ)
        a.run(messages=self.FIRST)
        prepared = a.app.prepare(dict(model='glm53', messages=self.SECOND, max_tokens=4096), True)
        with self.assertRaises(RequestCancelled):
            server.cached_prefill(a.runtime, prepared.prompt, prepared.sampling, False, True, lambda: True)
        self.assertEqual(a.filled[-1], 0)
        self.assertEqual(server.prefix_states(a).snaps, [])
        self.assertEqual(a.run(messages=self.SECOND)['cached_tokens'], 0)

    def test_image_prompt_is_neither_resumed_nor_kept(self):
        a = CPUModel(self.SEQ)
        a.run(messages=self.FIRST)
        prepared = a.app.prepare(dict(model='glm53', messages=self.SECOND, max_tokens=4096), True)
        a.w.mtp = None                                                    # an image prompt fills without the MTP head
        first, cached = server.cached_prefill(a.runtime, prepared.prompt, prepared.sampling, True, False, None)
        self.assertEqual(cached, 0)
        self.assertEqual(server.prefix_states(a).snaps, [])


class KeepAlive(unittest.TestCase):
    """v57: a silent queued or prefilling stream gets empty chunks; a prompt whose client left is abandoned."""

    SEQ = [CLOSE] + enc('{"wire":"ok"}') + [EOS[0]]
    BODY = dict(model='glm53', messages=[dict(role='user', content='object')], max_tokens=4096,
                response_format={'type': 'json_object'}, stream=True)

    def served(self, delay):
        from tensorfold.cuda.http import make_handler
        f = CPUModel(self.SEQ)
        f.app.keepalive_seconds = 0.05
        fill, f.started, f.finished = f.api.prefill, threading.Event(), threading.Event()

        def slow(*args, **kwargs):
            try:
                f.started.set()
                time.sleep(delay)
                return fill(*args, **kwargs)
            finally:
                f.finished.set()
        f.api.prefill = slow
        httpd = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(f.app))
        worker = threading.Thread(target=httpd.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
        worker.start()
        self.addCleanup(worker.join)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        return f, httpd.server_port

    def test_silent_prompt_gets_empty_chunks_and_the_reply_is_unchanged(self):
        f, port = self.served(0.5)
        conn = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
        try:
            conn.request('POST', '/v1/chat/completions', json.dumps(self.BODY), {'Content-Type': 'application/json'})
            response = conn.getresponse()
            text = response.read().decode()
        finally:
            conn.close()
        self.assertEqual(response.status, 200)
        self.assertTrue(text.rstrip().endswith('data: [DONE]'))
        events = [json.loads(line[6:]) for line in text.splitlines() if line.startswith('data: {')]
        empty = [i for i, e in enumerate(events)
                 if e['choices'][0]['delta'] == {} and e['choices'][0]['finish_reason'] is None]
        self.assertGreaterEqual(len(empty), 3)
        content = ''.join(e['choices'][0]['delta'].get('content', '') for e in events)
        self.assertEqual(json.loads(content), {'wire': 'ok'})
        self.assertEqual(events[-1]['usage']['completion_tokens'], len(self.SEQ))
        self.assertNotIn(len(events) - 1, empty)                          # nothing follows the reply's last chunk

    def test_no_empty_chunks_when_keepalive_is_off(self):
        f, port = self.served(0.3)
        f.app.keepalive_seconds = 0
        conn = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
        try:
            conn.request('POST', '/v1/chat/completions', json.dumps(self.BODY), {'Content-Type': 'application/json'})
            text = conn.getresponse().read().decode()
        finally:
            conn.close()
        events = [json.loads(line[6:]) for line in text.splitlines() if line.startswith('data: {')]
        self.assertFalse([e for e in events if e['choices'][0]['delta'] == {} and e['choices'][0]['finish_reason'] is None])
        self.assertEqual(json.loads(''.join(e['choices'][0]['delta'].get('content', '') for e in events)), {'wire': 'ok'})

    def test_prompt_of_a_client_that_left_is_abandoned_before_it_computes(self):
        f, port = self.served(0.6)
        payload = json.dumps(self.BODY).encode()
        sock = socket.create_connection(('127.0.0.1', port), timeout=10)
        sock.sendall(b'POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n'
                     b'Content-Length: ' + str(len(payload)).encode() + b'\r\n\r\n' + payload)
        self.assertIn(b'200', sock.recv(4096))
        self.assertTrue(f.started.wait(10))
        sock.close()                                                       # the client leaves during the prompt
        self.assertTrue(f.finished.wait(10))
        self.assertEqual(f.prefills, 1)
        self.assertEqual(f.filled, [0])
        self.assertEqual(f.history, [])
        self.assertEqual(server.prefix_states(f).snaps, [])


if __name__ == '__main__':
    print(json.dumps({'kind': 'v54_cpu_environment', 'versions': {n: importlib.metadata.version(n) for n in
          ('torch', 'xgrammar', 'transformers', 'tokenizers')}, 'tokenizer_sha256': hashlib.sha256((MODEL / 'tokenizer.json').read_bytes()).hexdigest(),
          'vocab': VOCAB, 'open': OPEN, 'close': CLOSE, 'eos': EOS, 'cuda_initialized': torch.cuda.is_initialized(),
          'numerical_forward': 'synthetic CPU; no model weights or CUDA kernels loaded'}), flush=True)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    print(json.dumps({'kind': 'v54_cpu_result', 'tests': result.testsRun, 'failures': len(result.failures),
          'errors': len(result.errors), 'cuda_initialized': torch.cuda.is_initialized()}), flush=True)
    sys.exit(not result.wasSuccessful())
