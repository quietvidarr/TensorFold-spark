"""CPU-only synthetic commit/alignment tests, not model qualification.

Execute the actual sampler, draft and absorb definitions without importing CUDA
kernels. Hidden rows encode the main-token position so shifted MTP pairing fails.
"""
import ast
import copy
import dataclasses
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import torch

COPY = Path(__file__).resolve().parents[1]
SOURCE = COPY / 'src/tensorfold/families/glm5_next/cuda/decode.py'
tree = ast.parse(SOURCE.read_text())
engine = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Engine')
functions = [next(n for n in engine.body if isinstance(n, ast.FunctionDef) and n.name == 'sample')]
functions += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('absorb', 'draft')]
unit = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + functions, type_ignores=[])
ast.fix_missing_locations(unit)
namespace = {'sample_rows': lambda w, y, p, s, offset, probs: y.argmax(dim=-1).tolist()}
exec(compile(unit, str(SOURCE), 'exec'), namespace)
spec = importlib.util.spec_from_file_location('mtp_test_serving', COPY / 'serving/server.py')
serving = importlib.util.module_from_spec(spec)
spec.loader.exec_module(serving)


class Landing:
    """Explicit state contract stub independent of the parallel state worker."""
    def __init__(self, trigger_at=None, cap=None, closed=False):
        self.config = NS(enabled=True)
        self.open_id, self.close_id = 1, 2
        self.closed, self.thinking = closed, not closed
        self.trigger = None
        self.trigger_at, self.cap = trigger_at, cap
        self.forced = []
        self.observed = []
        self.logger = None
        self.termination_reason = None
    @property
    def mask_close(self):
        return self.closed
    @property
    def serial_required(self):
        return bool(self.forced) or (self.trigger is not None and not self.closed)
    @property
    def should_stop(self):
        return self.termination_reason is not None
    def next_forced(self):
        return self.forced[0] if self.forced else None
    def close_bias(self):
        return 0.0
    def simulation(self):
        result = copy.deepcopy(self)
        result.logger = None
        return result
    def observe(self, token, piece=''):
        self.observed.append(token)
        if self.forced and token == self.forced[0]:
            self.forced.pop(0)
        if token == self.close_id:
            self.closed, self.thinking = True, False
        if self.trigger_at == len(self.observed):
            self.trigger = 'budget'
            self.forced = [3, 4, 2, 11, 12, 13]
        if self.cap == len(self.observed):
            self.termination_reason = 'answer_cap'
    def metadata(self):
        return dict(total_emitted=len(self.observed), termination_reason=self.termination_reason)


@dataclasses.dataclass
class Sampling:
    temperature: float = 0.0


class Fake:
    def __init__(self, sequence, landing, reject=False):
        self.sequence = sequence
        self.landing = landing
        self.reject = reject
        self.w = NS(cfg=NS(eos=(9,)), vocab_offset=0)
        self.st = NS(pos=0, capacity=5000, mtp_len=0, mtp_drafted=0)
        self.st.set_mtp_len = lambda n: setattr(self.st, 'mtp_len', n)
        self.buf = None
        self.mbuf = NS(rows=2)
        self.graphs = None
        self.replays = {}
        self.constraint = self.window = self.thinking_landing = None
        self.context, self.history, self.commits, self.mtp_pairs, self.samples = [], [], [], [], []
        self.draft_calls = []
        def draft(e, hidden, tokens, position, count, sampling):
            self.draft_calls.append((count, self.thinking_landing.serial_required))
            return namespace['draft'](e, hidden, tokens, position, count, sampling)
        self.decode_piece = lambda token: str(token)
        self.rows = None
        self.last_hidden = torch.zeros(1, 1)
        self.scratch = torch.empty(2, 1)
        self.api = NS(prefill=self.prefill, commit=self.commit, draft=draft,
                      take_snapshot=lambda e,ids,pending,**kw:NS(ids=list(ids),mtp_len=-1))  # never resumed by these doubles
    def reset(self):
        self.st.pos = self.st.mtp_len = self.st.mtp_drafted = 0
    def logits(self, token):
        row = torch.zeros(1, 32)
        row[0, token] = 20
        return row
    def wanted(self, index):
        return self.sequence[index] if index < len(self.sequence) else 9
    def prefill(self, e, prompt, sampling, **kwargs):
        self.context = list(prompt)
        self.st.pos = len(prompt)
        self.st.mtp_len = len(prompt) - 1
        self.last_hidden.fill_(len(prompt) - 1)
        first = self.sample(self.logits(self.wanted(0)), [len(prompt)], sampling)[0]
        self.follow([first])
        return first
    def verify_window(self, tokens):
        return tokens
    def forward(self, tokens):
        self.rows = list(tokens)
        self.hiddens = torch.arange(self.st.pos, self.st.pos + len(tokens), dtype=torch.float32).view(-1, 1)
        self.graph_output = torch.cat([self.logits(self.wanted(len(self.history) + i)) for i in range(len(tokens))])
        self.graph_original = self.graph_output.clone()
        return self.graph_output
    def sample(self, y, positions, sampling, **kwargs):
        self.samples.append((kwargs.get('draft', False), tuple(self.thinking_landing.observed)))
        return namespace['sample'](self, y, positions, sampling, **kwargs)
    def mtp(self, tokens, hidden):
        for i, token in enumerate(tokens):
            position = self.st.mtp_len + i
            if int(hidden[i, 0]) != position:
                raise AssertionError(('MTP hidden alignment', position, hidden.tolist(), tokens))
            self.mtp_pairs.append((position, token))
        return self.logits(31 if self.reject else self.wanted(len(self.history)))
    def main_hidden(self, rows):
        return self.hiddens[rows]
    def follow(self, tokens):
        self.history.extend(tokens)
    def commit(self, w, st, b, rows, keep):
        self.commits.append((rows, keep))
        self.context.extend(self.rows[:keep])
        st.pos += keep
        assert torch.equal(self.graph_output, self.graph_original), 'graph output mutated'
    def run(self, count, draft=True, callback=None, greedy_none=False):
        cuda = NS(synchronize=lambda: None, max_memory_reserved=lambda: 0)
        runtime = (self, NS(decode=lambda ids, **kw: self.decode_piece(ids[0])), NS(inference_mode=torch.inference_mode, cuda=cuda), self.api, Sampling, self.scratch, None, NS(guard=lambda phase: None), {})
        self.out = []
        def emit(tokens):
            self.out.extend(tokens)
            return bool(callback(tokens)) if callback else False
        self.stats = serving.generate(runtime, [20, 1], count, None if greedy_none else Sampling(), emit, draft=draft, landing_state=self.landing)
        return self.stats


from tensorfold.engine.thinking_landing import LandingConfig, ThinkingLanding


class TrackedLanding(ThinkingLanding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observed = []
    def observe(self, token, piece=''):
        super().observe(token, piece)
        self.observed.append(token)


def real_state(**options):
    return TrackedLanding(LandingConfig(budget=2, grace_tokens=0, bias_max=0,
                                       repeat_count=10000, variety_fraction=0.001,
                                       **options), 1, 2, [3, 4], 4096,
                          answer_prefix_ids=[11, 12, 13])


class MTPTests(unittest.TestCase):
    def test_greedy_none_sampler_stats_are_serializable(self):
        fake = Fake([5, 6, 2, 8, 9], Landing())
        stats = fake.run(16, greedy_none=True)
        self.assertIsNone(stats['sampling'])
        self.assertEqual(fake.out, [5, 6, 2, 8, 9])

    def test_serving_constructor_has_no_forced_answer_heading(self):
        server_tree = ast.parse((COPY / 'serving/server.py').read_text())
        call = next(n for n in ast.walk(server_tree) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id == 'ThinkingLanding')
        encodings = []
        def encode(text, add_special_tokens):
            self.assertFalse(add_special_tokens)
            encodings.append(text)
            return NS(ids=[565, 21754, 271] if text == '## Answer\n\n' else [3, 4])
        tok = NS(encode=encode, token_to_id=lambda t: {'<think>': 1, '</think>': 2, '<|assistant|>': 10}[t])
        namespace = dict(ThinkingLanding=ThinkingLanding, tok=tok,
                         prepared=NS(max_tokens=4096, prompt=[10, 1]),
                         config=LandingConfig(), initial_thinking=serving.initial_thinking,
                         LANDING_TEXT=serving.LANDING_TEXT)
        state = eval(compile(ast.Expression(call), 'serving constructor', 'eval'), namespace)
        self.assertEqual(state.answer_prefix_ids, ())
        self.assertEqual(encodings, [serving.LANDING_TEXT])
    def test_second_target_row_uses_new_close_bias(self):
        class BiasLanding(Landing):
            def close_bias(self):
                return 25.0 if self.thinking and len(self.observed) >= 2 else 0.0
        fake = Fake([5, 6, 7, 8, 9], BiasLanding())
        fake.run(32)
        self.assertEqual(fake.out, [5, 6, 2, 8, 9])
        self.assertEqual(fake.commits[0], (2, 2))
    def test_real_answer_repeat_stops_first_row_and_logs_once(self):
        state = TrackedLanding(LandingConfig(budget=2, grace_tokens=0, bias_max=0,
                                            repeat_ngram=1, repeat_count=3,
                                            repeat_min_thinking_tokens=0),
                               1, 2, [3, 4], 4096, answer_prefix_ids=[11, 12, 13])
        events = []
        state.logger = events.append
        fake = Fake([5, 6] + [7] * 40, state)
        fake.decode_piece = lambda token: 'word'
        fake.run(4096)
        self.assertEqual(fake.out, [5, 6, 3, 4, 2, 11, 12, 13, 7, 7, 7])
        self.assertEqual(fake.commits[-1], (2, 1))
        self.assertEqual(fake.stats['stop_reason'], 'answer_repeat_ngram')
        self.assertEqual(len(events), 2)  # landing + real termination, not simulation
        self.assertEqual(state.observed, fake.out)
    def test_grammar_pruned_proposals_still_count_as_drafted(self):
        fake = Fake([5, 6, 7, 8, 9], Landing(closed=True))
        fake.verify_window = lambda tokens: tokens[:1]
        fake.run(32)
        self.assertEqual(fake.stats['drafted'], 4)
        self.assertEqual(fake.stats['accepted'], 0)
        self.assertEqual(fake.stats['serial_rounds'], 0)

    def test_real_natural_close_during_trigger_grace_resumes_mtp(self):
        state = TrackedLanding(LandingConfig(budget=1, grace_tokens=2, bias_max=0),
                               1, 2, [3, 4], 4096, answer_prefix_ids=[11, 12, 13])
        fake = Fake([5, 2, 8, 9], state)
        fake.run(4096)
        self.assertEqual(fake.out, [5, 2, 8, 9])
        self.assertFalse(state.landed)
        self.assertEqual(fake.commits, [(1, 1), (2, 2)])
        self.assertEqual(fake.stats['mtp_resumes'], 1)
        self.assertEqual(fake.stats['accepted'], 1)
        self.assertEqual(state.observed, fake.out)

    def test_real_landing_trigger_resumes_and_assimilates_prefix(self):
        state = real_state()
        fake = Fake([5, 6, 7, 7, 7, 7, 7, 7, 8, 9], state)
        fake.run(4096)
        self.assertEqual(fake.out, [5, 6, 3, 4, 2, 11, 12, 13, 8, 9])
        self.assertEqual(fake.commits[0], (2, 1))
        self.assertEqual(state.observed, fake.out)
        self.assertEqual(state.answer_count, 5)
        self.assertEqual(fake.stats['mtp_resumes'], 1)
        self.assertTrue(all(count == 0 for count, serial in fake.draft_calls if serial))
        self.assertEqual(fake.stats['drafted'], sum(count for count, serial in fake.draft_calls))
        self.assertEqual(fake.mtp_pairs, [(i + 1, t) for i, t in enumerate(fake.out[:-2])])
        self.assertEqual(fake.context, [20, 1] + fake.out[:-1])
    def test_real_answer_cap_includes_prefix_without_overshoot(self):
        state = real_state()
        fake = Fake([5, 6] + [7] * 4094, state)
        fake.run(4096)
        self.assertEqual(state.answer_count, 2048)
        self.assertEqual(len(fake.out), 2053)
        self.assertEqual(fake.out[5:8], [11, 12, 13])
        self.assertEqual(state.total_emitted, len(fake.out))
        self.assertEqual(fake.stats['stop_reason'], 'answer_cap')
        self.assertEqual(state.observed, fake.out)
    def test_real_zero_budget_prefill_forces_close_then_header(self):
        state = TrackedLanding(LandingConfig(budget=0), 1, 2, [3, 4], 2048,
                               answer_prefix_ids=[11, 12, 13])
        fake = Fake([5, 6, 7, 7, 8, 9], state)
        fake.run(2048)
        self.assertEqual(fake.out, [2, 11, 12, 13, 8, 9])
        self.assertEqual(state.thinking_count, 0)
        self.assertEqual(fake.context, [20, 1] + fake.out[:-1])
        self.assertEqual(fake.stats['mtp_resumes'], 1)
    def test_real_second_row_answer_cap_uses_updated_snapshot(self):
        state = real_state(answer_max_tokens=5)
        fake = Fake([5, 6, 7, 7, 7, 7, 7, 7, 8, 10, 14], state)
        fake.run(4096)
        self.assertEqual(state.answer_count, 5)
        self.assertEqual(fake.out, [5, 6, 3, 4, 2, 11, 12, 13, 8, 10])
        self.assertEqual(fake.commits[-1], (2, 2))
        self.assertEqual(fake.stats['stop_reason'], 'answer_cap')
    def test_target_only_grammar_masks_full_window_without_buffer_changes(self):
        fake = Fake([5, 6, 7, 2, 9], Landing())
        calls = []
        def verify(tokens):
            fake.window = NS(tokens=tokens)
            return tokens
        def mask(logits, window, offset):
            calls.append(tuple(logits.shape))
            self.assertEqual(logits.shape[0], len(window.tokens))
            if logits.shape[0] == 2:
                logits[1, 7] = -float('inf')
                logits[1, 8] = 30
        fake.verify_window = verify
        fake.constraint = NS(mask=mask)
        fake.run(32)
        self.assertEqual(fake.out, [5, 6, 8, 2, 9])
        self.assertEqual(calls[0], (2, 32))
        self.assertIsNone(fake.window)
    def test_second_row_sample_failure_restores_real_state(self):
        fake = Fake([5, 6, 7], Landing())
        original = fake.sample
        def sample(y, positions, sampling, **kwargs):
            if not kwargs.get('draft') and fake.thinking_landing is not fake.landing:
                raise RuntimeError('synthetic row-one sample failure')
            return original(y, positions, sampling, **kwargs)
        fake.sample = sample
        with self.assertRaisesRegex(RuntimeError, 'row-one sample failure'):
            fake.run(32)
        self.assertEqual(fake.landing.observed, [5])
        self.assertEqual(fake.commits, [])
        self.assertIsNone(fake.thinking_landing)

    def test_trigger_first_candidate_cuts_window_and_resumes_after_header(self):
        state = Landing(trigger_at=2)
        fake = Fake([5, 6, 7, 7, 7, 7, 7, 7, 8, 9], state)
        fake.run(32)
        self.assertEqual(fake.out, [5, 6, 3, 4, 2, 11, 12, 13, 8, 9])
        self.assertEqual(fake.commits[0], (2, 1))
        self.assertEqual(state.observed, fake.out)
        self.assertEqual(fake.context, [20, 1] + fake.out[:-1])
        self.assertEqual(fake.stats['mtp_resumes'], 1)
        self.assertGreater(fake.stats['serial_rounds'], 0)
        self.assertGreater(fake.stats['drafted'], 1)
        self.assertIsNone(fake.thinking_landing)
    def test_rejected_proposals_never_reach_observer(self):
        state = Landing()
        fake = Fake([5, 6, 7, 8, 9], state, reject=True)
        state.closed, state.thinking = True, False
        fake.run(32)
        self.assertEqual(state.observed, [5, 6, 7, 8, 9])
        self.assertNotIn(31, state.observed)
        self.assertEqual(fake.stats['accepted'], 0)
        self.assertTrue(all(keep == 1 for rows, keep in fake.commits))
    def test_no_trigger_serial_and_mtp_observe_identical_tokens(self):
        serial = Fake([5, 6, 7, 2, 8, 9], Landing())
        mtp = Fake([5, 6, 7, 2, 8, 9], Landing())
        serial.run(32, draft=False)
        mtp.run(32)
        self.assertEqual(serial.out, mtp.out)
        self.assertEqual(serial.landing.observed, mtp.landing.observed)
        self.assertGreater(mtp.stats['accepted'], 0)
        # Target row 1 sees a simulation including accepted row 0, not the proposal.
        self.assertIn((False, (5, 6)), mtp.samples)
    def test_natural_close_cuts_first_window(self):
        fake = Fake([5, 2, 8, 9], Landing())
        fake.run(32)
        self.assertEqual(fake.commits[0], (2, 1))
        self.assertEqual(fake.out, [5, 2, 8, 9])
    def test_stop_first_candidate_does_not_commit_second(self):
        fake = Fake([5, 6, 7, 8], Landing(cap=2))
        fake.run(32)
        self.assertEqual(fake.out, [5, 6])
        self.assertEqual(fake.commits, [(2, 1)])
        self.assertEqual(fake.stats['stop_reason'], 'answer_cap')
    def test_stop_second_candidate_and_callback_observe_entire_commit(self):
        fake = Fake([5, 6, 7, 8], Landing(cap=3))
        fake.run(32, callback=lambda ts: len(ts) == 2)
        self.assertEqual(fake.out, [5, 6, 7])
        self.assertEqual(fake.landing.observed, fake.out)
        self.assertEqual(fake.commits, [(2, 2)])
    def test_generation_count_never_overshoots(self):
        for count in (1, 2, 3, 4, 5):
            fake = Fake([5, 6, 7, 8, 10, 14], Landing())
            fake.run(count)
            self.assertEqual(len(fake.out), count)
            self.assertEqual(fake.landing.observed, fake.out)
    def test_draft_never_forces_or_observes(self):
        fake = Fake([5], Landing())
        fake.landing.forced = [3]
        fake.thinking_landing = fake.landing
        before = copy.deepcopy(fake.landing.__dict__)
        y = fake.logits(7)
        self.assertEqual(fake.sample(y, [1], Sampling(), draft=True), [7])
        self.assertEqual(fake.landing.__dict__, before)
    def test_closed_masks_both_delimiters_without_mutating_buffer(self):
        fake = Fake([5], Landing(closed=True))
        fake.thinking_landing = fake.landing
        y = fake.logits(2)
        y[0, 1] = 30
        y[0, 8] = 10
        original = y.clone()
        self.assertEqual(fake.sample(y, [1], Sampling(), draft=True), [8])
        self.assertTrue(torch.equal(y, original))
    def test_exception_restores_request_state(self):
        fake = Fake([5, 6, 7], Landing())
        old = object()
        fake.thinking_landing = old
        def fail(*args):
            raise RuntimeError('synthetic commit failure')
        fake.api.commit = fail
        with self.assertRaisesRegex(RuntimeError, 'synthetic commit failure'):
            fake.run(32)
        self.assertIs(fake.thinking_landing, old)
        self.assertEqual(fake.landing.observed, [5])


if __name__ == '__main__':
    unittest.main()
