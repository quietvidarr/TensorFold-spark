"""CPU-only synthetic and captured-token tests, without package/GPU imports."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "src/tensorfold/engine/thinking_landing.py"
_SPEC = importlib.util.spec_from_file_location("_cpu_thinking_landing", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
LandingConfig = _MODULE.LandingConfig
ThinkingLanding = _MODULE.ThinkingLanding

OPEN, CLOSE = 90001, 90002
PHRASE = [80001, 80002, 80003]


def state(config=None, **kwargs):
    return ThinkingLanding(config or LandingConfig(), OPEN, CLOSE, PHRASE,
                           kwargs.pop("output_cap", 12000), **kwargs)


def drain(machine):
    result = []
    while machine.next_forced() is not None:
        token = machine.next_forced()
        result.append(token)
        machine.observe(token, "synthetic")
    return result


class ConfigTests(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(LandingConfig(), LandingConfig.from_request({}))
        self.assertEqual(LandingConfig().budget, 4000)
        self.assertEqual(LandingConfig().answer_max_tokens, 2048)
        self.assertEqual(LandingConfig().repeat_min_thinking_tokens, 1500)
        self.assertEqual(LandingConfig().repeat_arithmetic_fraction, .75)

    def test_request_every_field_and_zero_override(self):
        values = dict(enabled=False, budget=44, repeat_ngram=2, repeat_count=10,
                      repeat_window=32, variety_window=16, variety_fraction=.125,
                      grace_tokens=0, answer_reserve=2048,
                      bias_start_fraction=.75, bias_max=3,
                      repeat_min_thinking_tokens=0, repeat_arithmetic_fraction=.9,
                      answer_max_tokens=1024)
        self.assertEqual(LandingConfig.from_request({"thinking_loop": values}),
                         LandingConfig(**values))
        result = LandingConfig.from_request(
            {"thinking_loop": {"budget": 22}, "thinking_budget": 0},
            LandingConfig(bias_max=5))
        self.assertEqual(result.budget, 0)
        self.assertEqual(result.bias_max, 5)

    def test_invalid_values(self):
        for values in [dict(enabled=1), dict(budget=True), dict(budget=-1),
                       dict(budget=1.5), dict(repeat_ngram=0),
                       dict(repeat_count=0), dict(repeat_window=0),
                       dict(repeat_ngram=10, repeat_window=9),
                       dict(variety_window=0), dict(variety_fraction=0),
                       dict(variety_fraction=1.1), dict(variety_fraction=float("nan")),
                       dict(variety_fraction=float("inf")), dict(grace_tokens=-1),
                       dict(grace_tokens=49), dict(answer_reserve=-1),
                       dict(answer_reserve=0), dict(answer_reserve=1023),
                       dict(bias_start_fraction=-.1), dict(bias_start_fraction=1),
                       dict(bias_max=-1), dict(bias_max=float("inf")),
                       dict(bias_max=True), dict(bias_start_fraction=".5")]:
            with self.subTest(values=values), self.assertRaises((ValueError, TypeError)):
                LandingConfig(**values)

    def test_invalid_request(self):
        for body in [[], {"thinking_loop": None}, {"thinking_loop": True},
                     {"thinking_loop": {"typo": 1}}, {"thinking_budget": "0"},
                     {"thinking_budget": False}]:
            with self.subTest(body=body), self.assertRaises((ValueError, TypeError)):
                LandingConfig.from_request(body)
        with self.assertRaises(TypeError):
            LandingConfig.from_request({}, defaults={})

    def test_repeat_request_overrides_and_inherited_defaults(self):
        defaults = LandingConfig(repeat_min_thinking_tokens=2000,
                                 repeat_arithmetic_fraction=.8)
        inherited = LandingConfig.from_request({}, defaults)
        self.assertEqual(inherited, defaults)
        overridden = LandingConfig.from_request({"thinking_loop": {
            "repeat_min_thinking_tokens": 0, "repeat_arithmetic_fraction": 1}}, defaults)
        self.assertEqual(overridden.repeat_min_thinking_tokens, 0)
        self.assertEqual(overridden.repeat_arithmetic_fraction, 1)

    def test_invalid_repeat_options(self):
        cases = [("repeat_min_thinking_tokens", value, TypeError)
                 for value in (True, False, 1500.0, "1500", None)]
        cases += [("repeat_min_thinking_tokens", -1, ValueError)]
        cases += [("repeat_arithmetic_fraction", value, TypeError)
                  for value in (True, False, ".75", None, complex(.75))]
        cases += [("repeat_arithmetic_fraction", value, ValueError)
                  for value in (0, -.1, 1.01, float("nan"),
                                float("inf"), float("-inf"))]
        for name, value, error in cases:
            with self.subTest(name=name, value=value):
                with self.assertRaises(error):
                    LandingConfig(**{name: value})
                with self.assertRaises(error):
                    LandingConfig.from_request({"thinking_loop": {name: value}})


class DetectorTests(unittest.TestCase):
    def test_repeated_ngram(self):
        machine = state(LandingConfig(repeat_min_thinking_tokens=0))
        for token in list(range(8)) * 3:
            machine.observe(token, "")
        self.assertEqual(machine.trigger, "repeat_ngram")
        self.assertEqual(machine.thinking_count, 24)

    def test_any_ngram_not_just_suffix(self):
        machine = state(LandingConfig(repeat_ngram=2, repeat_count=3,
                                      variety_window=1000, repeat_min_thinking_tokens=0))
        for token in [1, 2, 9, 1, 2, 8, 1, 2]:
            machine.observe(token, "")
        self.assertEqual(machine.trigger, "repeat_ngram")

    def test_repeat_window_eviction(self):
        machine = state(LandingConfig(repeat_ngram=2, repeat_count=3,
                                      repeat_window=5, variety_window=1000,
                                      repeat_min_thinking_tokens=0))
        for token in [1, 2, 3, 1, 2, 4, 1, 2, 5]:
            machine.observe(token, "")
        self.assertIsNone(machine.trigger)

    def test_low_variety_requires_full_window(self):
        machine = state(LandingConfig(repeat_count=100000, repeat_min_thinking_tokens=0))
        for i in range(255):
            machine.observe(i % 4)
        self.assertIsNone(machine.trigger)
        machine.observe(3)
        self.assertEqual(machine.trigger, "low_variety")

    def test_variety_threshold_is_strict(self):
        machine = state(LandingConfig(repeat_count=100000, repeat_min_thinking_tokens=0))
        for i in range(256):
            machine.observe(i % 64)
        self.assertIsNone(machine.trigger)

    def test_budget_without_other_detectors(self):
        machine = state(LandingConfig(budget=5, repeat_window=100,
                                      variety_window=100))
        for token in range(4):
            machine.observe(token)
        self.assertIsNone(machine.trigger)
        machine.observe(4)
        self.assertEqual(machine.trigger, "budget")

    def test_8000_unique_tokens_no_trigger(self):
        machine = state(LandingConfig(budget=10000), output_cap=14000)
        for token in range(8000):
            machine.observe(token)
        self.assertIsNone(machine.trigger)
        self.assertEqual(machine.thinking_count, 8000)

    def test_disabled(self):
        machine = state(LandingConfig(enabled=False, budget=0), output_cap=1)
        for _ in range(800):
            machine.observe(1)
        self.assertIsNone(machine.trigger)
        self.assertIsNone(machine.next_forced())
        self.assertEqual(machine.close_bias(), 0)

    def test_outside_thinking(self):
        machine = state(initial_thinking=False)
        for _ in range(800):
            machine.observe(1)
        self.assertFalse(machine.thinking)
        self.assertEqual(machine.thinking_count, 0)
        self.assertIsNone(machine.trigger)
        machine.observe(OPEN)
        for token in range(8):
            machine.observe(token)
        self.assertIsNone(machine.trigger)
        self.assertEqual(machine.thinking_count, 8)


class RepeatReplayHelpers:
    SENTENCE = [" We", " should", " verify", " each", " assumption",
                " before", " giving", " answers."]

    @staticmethod
    def prelude(machine, count=1500):
        for token in range(100000, 100000 + count):
            machine.observe(token, " unique")

    def repeat(self, machine, pieces, ids=tuple(range(8)), times=3):
        self.assertEqual(len(ids), 8)
        self.assertEqual(len(pieces), 8)
        for _ in range(times):
            for token, piece in zip(ids, pieces, strict=True):
                machine.observe(token, piece)


class RepeatRetuningTests(RepeatReplayHelpers, unittest.TestCase):
    def test_before_arming_no_repeat(self):
        machine = state()
        self.prelude(machine, 1475)
        self.repeat(machine, self.SENTENCE)
        self.assertEqual(machine.thinking_count, 1499)
        self.assertIsNone(machine.trigger)
        self.assertIsNone(machine.next_forced())

    def test_arming_uses_occurrences_accumulated_in_window(self):
        machine = state()
        self.prelude(machine, 1476)
        self.repeat(machine, self.SENTENCE)
        self.assertEqual(machine.trigger, "repeat_ngram")
        self.assertEqual(machine.metadata()["trigger_thinking_count"], 1500)
        self.assertEqual(machine.next_forced(), PHRASE[0])

    def test_meaningful_sentence_after_arming_triggers(self):
        machine = state()
        self.prelude(machine)
        self.repeat(machine, self.SENTENCE, times=2)
        self.assertIsNone(machine.trigger)
        self.repeat(machine, self.SENTENCE, times=1)
        self.assertEqual(machine.trigger, "repeat_ngram")
        self.assertEqual(machine.metadata()["trigger_thinking_count"], 1524)

    def test_old_occurrences_expire_before_arming(self):
        machine = state()
        self.repeat(machine, self.SENTENCE)
        self.prelude(machine)
        self.repeat(machine, self.SENTENCE, times=2)
        self.assertIsNone(machine.trigger)
        self.repeat(machine, self.SENTENCE, times=1)
        self.assertEqual(machine.trigger, "repeat_ngram")

    def test_unicode_arithmetic_character_classes_and_threshold(self):
        cases = [(["１", "²", "Ⅷ", "a", "", "", "", ""], None),
                 (["（", "）", "。", "a", "", "", "", ""], None),
                 (["∑", "√", "∞", "a", "", "", "", ""], None),
                 (["^", "^", "^", "a", "", "", "", ""], None),
                 ([" １\t", "²\n", "Ⅷ ", "a", "", "", "", ""], None),
                 (["１", "²", "a", "", "", "", "", ""], "repeat_ngram"),
                 (["$", "$", "$", "a", "", "", "", ""], "repeat_ngram"),
                 ([""] * 8, "repeat_ngram"),
                 ([" \t"] * 8, "repeat_ngram")]
        for pieces, expected in cases:
            with self.subTest(pieces=pieces):
                machine = state(LandingConfig(repeat_min_thinking_tokens=0))
                self.repeat(machine, pieces)
                self.assertEqual(machine.trigger, expected)

    def test_default_512_token_window_boundary(self):
        for gap, expected in [(488, "repeat_ngram"), (489, None)]:
            with self.subTest(gap=gap):
                machine = state()
                self.prelude(machine)
                self.repeat(machine, self.SENTENCE, times=1)
                for token in range(200000, 200000 + gap):
                    machine.observe(token, " unique")
                self.repeat(machine, self.SENTENCE, times=2)
                self.assertEqual(machine.trigger, expected)

    def test_arithmetic_fraction_weights_characters_not_pieces(self):
        cases = [(["123456789012345678901234", "a", "b", "c", "d", "e", "f", "g"], None),
                 (["1", "2", "3", "4", "5", "6", "7", "a long meaningful word"],
                  "repeat_ngram")]
        for pieces, expected in cases:
            with self.subTest(pieces=pieces):
                machine = state(LandingConfig(repeat_min_thinking_tokens=0))
                self.repeat(machine, pieces)
                self.assertEqual(machine.trigger, expected)

    def test_classifier_uses_current_candidate_pieces(self):
        machine = state()
        self.prelude(machine)
        self.repeat(machine, [str(token) for token in range(8)], times=2)
        self.repeat(machine, self.SENTENCE, times=1)
        self.assertEqual(machine.trigger, "repeat_ngram")
        self.assertEqual(machine.metadata()["trigger_thinking_count"], 1524)

    def test_arithmetic_fraction_override(self):
        pieces = ["1", "2", "3", "a", "", "", "", ""]
        for fraction, expected in [(.75, None), (1, "repeat_ngram")]:
            with self.subTest(fraction=fraction):
                machine = state(LandingConfig(repeat_min_thinking_tokens=0,
                                              repeat_arithmetic_fraction=fraction))
                self.repeat(machine, pieces)
                self.assertEqual(machine.trigger, expected)

    def test_arithmetic_low_variety_has_zero_eligible_tokens(self):
        machine = state(LandingConfig(repeat_min_thinking_tokens=0))
        for index in range(1800):
            machine.observe(index % 8, str(index % 8))
        self.assertIsNone(machine.trigger)
        self.assertEqual(machine._variety_eligible, 0)
        self.assertEqual(len(machine._variety_tokens), 256)
        self.assertEqual(len(machine._variety_counts), 0)

    def test_repeat_has_priority_over_low_variety_when_armed(self):
        machine = state(LandingConfig(repeat_min_thinking_tokens=0,
                                      variety_window=24, variety_fraction=.5))
        self.repeat(machine, self.SENTENCE)
        self.assertEqual(machine.trigger, "repeat_ngram")

    def test_budget_has_priority_over_armed_repeat(self):
        machine = state(LandingConfig(budget=24, repeat_min_thinking_tokens=0,
                                      variety_window=24, variety_fraction=.5))
        self.repeat(machine, self.SENTENCE)
        self.assertEqual(machine.trigger, "budget")
        self.assertEqual(machine.metadata()["trigger_thinking_count"], 24)


class VarietyRetuningTests(RepeatReplayHelpers, unittest.TestCase):
    def test_default_gate_arms_low_variety_at_1500_using_existing_window(self):
        machine = state(LandingConfig(repeat_count=100000))
        for index in range(1499):
            machine.observe(index % 4, " prose")
            self.assertIsNone(machine.trigger)
        machine.observe(3, " prose")
        self.assertEqual(machine.trigger, "low_variety")
        self.assertEqual(machine.metadata()["trigger_thinking_count"], 1500)

    def test_long_nonarithmetic_loop_after_1500_triggers_low_variety(self):
        machine = state(LandingConfig(repeat_count=100000))
        self.prelude(machine)
        for index in range(256):
            machine.observe(index % 4, " prose")
        self.assertEqual(machine.trigger, "low_variety")
        self.assertGreater(machine.metadata()["trigger_thinking_count"], 1500)

    def test_raw_window_not_filtered_window_and_eligible_denominator(self):
        # 248 repeated numeric tokens are excluded. Eight eligible tokens
        # with two distinct IDs have ratio 2/8 == .25, not 2/256.
        machine = state(LandingConfig(repeat_count=100000, repeat_min_thinking_tokens=0))
        for index in range(248):
            machine.observe(99, "123")
        for index in range(8):
            machine.observe(index % 2, " word")
        self.assertIsNone(machine.trigger)
        self.assertEqual(machine._variety_eligible, 8)
        self.assertEqual(len(machine._variety_counts), 2)
        # Evict a numeric token, admitting one more prose token: 2/9 < .25.
        machine.observe(0, " word")
        self.assertEqual(machine.trigger, "low_variety")
        self.assertEqual(machine.metadata()["trigger_thinking_count"], 257)

    def test_partial_raw_window_never_fires_even_with_low_eligible_ratio(self):
        machine = state(LandingConfig(repeat_count=100000, repeat_min_thinking_tokens=0))
        for index in range(255):
            machine.observe(1, " word" if index < 64 else "123")
        self.assertIsNone(machine.trigger)
        self.assertEqual(machine._variety_eligible, 64)
        machine.observe(1, "123")
        self.assertEqual(machine.trigger, "low_variety")

    def test_arithmetic_piece_classifier_threshold_unicode_and_blank(self):
        for piece, excluded in [("１²Ⅷa", True), ("（）。a", True),
                                ("∑√∞a", True), ("^^^a", True),
                                (" \t１²Ⅷa\n", True), ("１２a", False),
                                ("$$$a", False), ("", False), (" \t", False),
                                ("+-*/%=^<>|~", True), ("_", True)]:
            with self.subTest(piece=piece):
                machine = state(LandingConfig(repeat_count=100000,
                                              repeat_min_thinking_tokens=0))
                for index in range(256):
                    machine.observe(index % 4, piece)
                self.assertEqual(machine.trigger, None if excluded else "low_variety")
                self.assertEqual(machine._variety_eligible, 0 if excluded else 256)

    def test_same_id_arithmetic_and_prose_occurrences_are_filtered_per_token(self):
        machine = state(LandingConfig(repeat_count=100000, repeat_min_thinking_tokens=0))
        machine.observe(1, " prose")
        for index in range(255):
            machine.observe(1, "1")
        self.assertIsNone(machine.trigger)  # one eligible token, ratio 1/1
        self.assertEqual(machine._variety_counts, {1: 1})
        machine.observe(1, "1")
        self.assertEqual(machine._variety_eligible, 0)
        self.assertEqual(machine._variety_counts, {})
        self.assertIsNone(machine.trigger)

    def test_classifier_fraction_override_applies_to_both_detectors(self):
        for fraction, expected in [(.75, None), (1, "low_variety")]:
            with self.subTest(fraction=fraction):
                machine = state(LandingConfig(repeat_count=100000,
                                              repeat_min_thinking_tokens=0,
                                              repeat_arithmetic_fraction=fraction))
                for index in range(256):
                    machine.observe(index % 4, "123a")
                self.assertEqual(machine.trigger, expected)


class V50CapturedRegressionTests(unittest.TestCase):
    def test_both_captured_seed_traces_with_and_without_common_gate(self):
        for filename, expected_count in [("v50_c_seed1.json", 609),
                                         ("v50_c_seed2.json", 306)]:
            fixture = json.loads((Path(__file__).parent / "fixtures" / filename)
                                 .read_text(encoding="utf-8"))
            ids = fixture["sampled_thinking_ids"]
            pieces = fixture["sampled_thinking_pieces"]
            self.assertEqual(len(ids), expected_count)
            self.assertEqual(len(pieces), expected_count)
            for minimum in (1500, 0):
                with self.subTest(fixture=filename, minimum=minimum):
                    machine = state(LandingConfig(repeat_min_thinking_tokens=minimum))
                    for index, (token, piece) in enumerate(zip(ids, pieces, strict=True), 1):
                        machine.observe(token, piece)
                        self.assertIsNone(machine.trigger,
                                          f"{filename}: fired at sampled token {index}")
                        self.assertIsNone(machine.next_forced())
                    self.assertEqual(machine.thinking_count, expected_count)
                    self.assertEqual(machine.metadata()["sampled_thinking_tokens"], expected_count)
                    self.assertEqual(machine.metadata()["forced_thinking_tokens"], 0)


class RealCRegressionTests(RepeatReplayHelpers, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads((Path(__file__).parent / "fixtures/c_on_arithmetic.json")
                                 .read_text(encoding="utf-8"))

    def test_real_532_sampled_tokens_never_trigger(self):
        ids = self.fixture["sampled_thinking_ids"]
        pieces = self.fixture["sampled_thinking_pieces"]
        self.assertEqual(len(ids), 532)
        self.assertEqual(len(pieces), 532)
        for minimum in (1500, 0):
            with self.subTest(repeat_min_thinking_tokens=minimum):
                machine = state(LandingConfig(repeat_min_thinking_tokens=minimum))
                for index, (token, piece) in enumerate(zip(ids, pieces, strict=True), 1):
                    machine.observe(token, piece)
                    self.assertIsNone(machine.trigger, f"detector fired at sampled token {index}")
                    self.assertIsNone(machine.next_forced())
                self.assertEqual(machine.metadata()["sampled_thinking_tokens"], 532)
                self.assertEqual(machine.metadata()["forced_thinking_tokens"], 0)

    def test_exact_real_arithmetic_ngram_after_arming_is_ignored(self):
        machine = state()
        self.prelude(machine)
        self.repeat(machine, self.fixture["ngram_pieces"], self.fixture["ngram_ids"])
        self.assertEqual(machine.thinking_count, 1524)
        self.assertIsNone(machine.trigger)
        self.assertIsNone(machine.next_forced())


    def test_arithmetic_filter_does_not_suppress_budget(self):
        machine = state(LandingConfig(budget=1524))
        self.prelude(machine)
        self.repeat(machine, self.fixture["ngram_pieces"], self.fixture["ngram_ids"])
        self.assertEqual(machine.trigger, "budget")
        self.assertEqual(machine.metadata()["trigger_thinking_count"], 1524)


class LandingTests(unittest.TestCase):
    def test_zero_budget_closes_empty_thinking_immediately(self):
        machine = state(LandingConfig(budget=0))
        self.assertEqual(machine.next_forced(), CLOSE)
        self.assertEqual(machine.next_forced(), CLOSE)  # peek, not pop
        self.assertEqual(list(machine.forced), [CLOSE])
        self.assertEqual(drain(machine), [CLOSE])
        self.assertEqual(machine.thinking_count, 0)
        self.assertTrue(machine.closed)
        self.assertTrue(machine.landed)

    def test_zero_budget_waits_until_open_when_initially_outside(self):
        machine = state(LandingConfig(budget=0), initial_thinking=False)
        self.assertIsNone(machine.next_forced())
        machine.observe(100)
        machine.observe(OPEN)
        self.assertEqual(drain(machine), [CLOSE])
        self.assertEqual(machine.thinking_count, 0)
        self.assertEqual(machine.total_emitted, 3)

    def test_reserve_lowers_budget(self):
        machine = state(output_cap=1100)
        self.assertEqual(machine.effective_budget, 24)

    def test_prior_prefix_accounted(self):
        machine = state(output_cap=1150, initial_thinking=False)
        for token in range(25):
            machine.observe(token)
        machine.observe(OPEN)
        self.assertEqual(machine.effective_budget, 48)

    def test_capacity_rejected_before_generation(self):
        with self.assertRaises(ValueError):
            state(output_cap=1027)
        state(output_cap=1028)  # phrase + close + reserve fit exactly
        with self.assertRaises(ValueError):
            state(LandingConfig(budget=0), output_cap=1024)
        self.assertEqual(state(LandingConfig(budget=0), output_cap=1025).next_forced(), CLOSE)

    def test_exact_landing_sequence_and_counting(self):
        machine = state(LandingConfig(budget=1))
        machine.observe(10, "done.")
        self.assertEqual(list(machine.forced), PHRASE + [CLOSE])
        self.assertEqual(drain(machine), PHRASE + [CLOSE])
        self.assertEqual(machine.total_emitted, 5)
        self.assertEqual(machine.thinking_count, 4)
        self.assertEqual(machine.metadata()["forced_thinking_tokens"], 3)
        self.assertEqual(machine.metadata()["sampled_thinking_tokens"], 1)
        self.assertTrue(machine.landed)

    def test_forced_phrase_does_not_run_detector(self):
        machine = ThinkingLanding(LandingConfig(budget=1, grace_tokens=0,
                                  repeat_ngram=1, repeat_count=2), OPEN, CLOSE,
                                  [10, 10, 10], 12000)
        machine.observe(11)
        self.assertEqual(drain(machine), [10, 10, 10, CLOSE])
        self.assertEqual(machine.trigger, "budget")
        self.assertEqual(machine.metadata()["forced_thinking_tokens"], 3)

    def test_grace_at_most_48_more_sampled_tokens(self):
        machine = state(LandingConfig(budget=1))
        machine.observe(10, "word")
        for i in range(47):
            machine.observe(i + 100, "word")
            self.assertIsNone(machine.next_forced())
        machine.observe(147, "word")
        self.assertEqual(machine.next_forced(), PHRASE[0])
        self.assertEqual(machine.thinking_count, 49)

    def test_newline_and_sentence_boundaries(self):
        for piece in ["line\n", "done.", "done!", "done?", 'done.\"']:
            with self.subTest(piece=piece):
                machine = state(LandingConfig(budget=1))
                machine.observe(10, piece)
                self.assertEqual(machine.next_forced(), PHRASE[0])

    def test_decimal_not_sentence_boundary(self):
        machine = state(LandingConfig(budget=1))
        machine.observe(10, "3.")
        self.assertIsNone(machine.next_forced())
        machine.observe(11, "14")
        self.assertIsNone(machine.next_forced())
        machine.observe(12, ".")
        self.assertIsNone(machine.next_forced())
        machine.observe(13, "\n")
        self.assertEqual(machine.next_forced(), PHRASE[0])

    def test_natural_close_during_grace(self):
        machine = state(LandingConfig(budget=1))
        machine.observe(10, "word")
        machine.observe(CLOSE)
        self.assertTrue(machine.closed)
        self.assertFalse(machine.landed)
        self.assertEqual(machine.trigger, "budget")
        self.assertIsNone(machine.next_forced())

    def test_natural_close_cancels_queued_phrase(self):
        machine = state(LandingConfig(budget=1, grace_tokens=0))
        machine.observe(10)
        machine.observe(CLOSE)
        self.assertFalse(machine.landed)
        self.assertEqual(list(machine.forced), [])

    def test_never_reopen_and_answer_mask_flag(self):
        machine = state()
        machine.observe(CLOSE)
        machine.observe(OPEN)
        machine.observe(10)
        self.assertFalse(machine.thinking)
        self.assertTrue(machine.closed)
        self.assertTrue(machine.metadata()["mask_open"])
        self.assertEqual(machine.thinking_count, 0)
        self.assertEqual(machine.close_bias(), 0)

    def test_bias_ramp(self):
        machine = state(LandingConfig(budget=100, variety_window=1000))
        self.assertEqual(machine.close_bias(), 0)
        for token in range(50):
            machine.observe(token)
        self.assertEqual(machine.close_bias(), 0)
        for token in range(50, 75):
            machine.observe(token)
        self.assertEqual(machine.close_bias(), 4)
        for token in range(75, 100):
            machine.observe(token)
        self.assertEqual(machine.close_bias(), 8)
        machine.observe(CLOSE)
        self.assertEqual(machine.close_bias(), 0)

    def test_log_only_after_actual_forced_close(self):
        logs = []
        machine = state(LandingConfig(budget=1, grace_tokens=0), logger=logs.append)
        machine.observe(10)
        self.assertEqual(logs, [])
        drain(machine)
        self.assertEqual(len(logs), 1)
        event = json.loads(logs[0])
        self.assertEqual(event["event"], "thinking_landing")
        self.assertEqual(event["trigger"], "budget")
        self.assertEqual(event["thinking_count"], 4)
        self.assertTrue(event["landed"])

    def test_natural_close_never_logs_landing(self):
        logs = []
        machine = state(LandingConfig(budget=1), logger=logs.append)
        machine.observe(10)
        machine.observe(CLOSE)
        self.assertEqual(logs, [])

    def test_callable_observes_emitted_token(self):
        machine = state()
        self.assertIsNone(machine(123, "piece"))
        self.assertEqual(machine.total_emitted, 1)
        self.assertEqual(machine.thinking_count, 1)

    def test_close_outside_thinking_is_terminal(self):
        machine = state(initial_thinking=False)
        machine.observe(CLOSE)
        machine.observe(OPEN)
        self.assertTrue(machine.closed)
        self.assertFalse(machine.thinking)
        self.assertFalse(machine.landed)
        self.assertTrue(machine.metadata()["mask_open"])

    def test_only_expected_forced_token_advances_queue(self):
        machine = state(LandingConfig(budget=1, grace_tokens=0))
        machine.observe(10)
        machine.observe(20)
        self.assertEqual(list(machine.forced), PHRASE + [CLOSE])
        machine.observe(PHRASE[0])
        self.assertEqual(list(machine.forced), PHRASE[1:] + [CLOSE])
        machine.observe(CLOSE)
        self.assertFalse(machine.landed)
        self.assertEqual(machine.metadata()["forced_thinking_tokens"], 1)

    def test_reserve_survives_maximum_grace_and_full_phrase(self):
        machine = state(output_cap=1100)
        for token in range(machine.effective_budget + 48):
            machine.observe(token, "word")
        self.assertEqual(drain(machine), PHRASE + [CLOSE])
        self.assertEqual(machine.output_cap - machine.total_emitted, 1024)

    def test_reserve_survives_prefix_and_maximum_grace(self):
        machine = state(output_cap=1150, initial_thinking=False)
        for token in range(25):
            machine.observe(token)
        machine.observe(OPEN)
        for token in range(machine.effective_budget + 48):
            machine.observe(token, "word")
        drain(machine)
        self.assertEqual(machine.output_cap - machine.total_emitted, 1024)

    def test_clamped_zero_still_reserves_phrase_and_answer(self):
        machine = state(output_cap=1028)
        self.assertEqual(machine.effective_budget, 0)
        self.assertEqual(drain(machine), PHRASE + [CLOSE])
        self.assertEqual(machine.output_cap - machine.total_emitted, 1024)

    def test_smaller_grace_and_custom_bias_ramp(self):
        machine = state(LandingConfig(budget=20, grace_tokens=2,
                                      bias_start_fraction=.25, bias_max=6))
        for token in range(5):
            machine.observe(token)
        self.assertEqual(machine.close_bias(), 0)
        for token in range(5, 10):
            machine.observe(token)
        self.assertEqual(machine.close_bias(), 2)
        for token in range(10, 20):
            machine.observe(token)
        self.assertEqual(machine.close_bias(), 6)
        machine.observe(20)
        self.assertIsNone(machine.next_forced())
        machine.observe(21)
        self.assertEqual(machine.next_forced(), PHRASE[0])

    def test_log_supports_standard_logger(self):
        class Logger:
            def __init__(self):
                self.messages = []

            def info(self, message):
                self.messages.append(message)

        logger = Logger()
        machine = state(LandingConfig(budget=0), logger=logger)
        drain(machine)
        event = json.loads(logger.messages[0])
        self.assertEqual(event["thinking_count"], 0)
        self.assertEqual(event["trigger_thinking_count"], 0)
        self.assertEqual(event["forced_thinking_tokens"], 0)

    def test_bad_constructor_and_observe_inputs(self):
        for kwargs in [dict(output_cap=True), dict(output_cap=0),
                       dict(open_id=True), dict(close_id=OPEN),
                       dict(landing_ids=[CLOSE]), dict(landing_ids=[OPEN]),
                       dict(initial_thinking=1), dict(logger=2)]:
            args = dict(config=LandingConfig(), open_id=OPEN, close_id=CLOSE,
                        landing_ids=PHRASE, output_cap=12000)
            args.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                ThinkingLanding(**args)
        machine = state()
        with self.assertRaises(TypeError):
            machine.observe(True)
        with self.assertRaises(TypeError):
            machine.observe(1, None)
        self.assertEqual(machine.total_emitted, 0)


class BudgetOnlyTests(RepeatReplayHelpers, unittest.TestCase):
    """v55: budget is the only thinking trigger and landed answers are unguarded."""

    def landed(self, **kwargs):
        machine = state(LandingConfig(budget=5, grace_tokens=0, thinking_triggers=False,
                                      **kwargs))
        for token in range(5):
            machine.observe(token, " word")
        self.assertEqual(machine.trigger, "budget")
        self.assertEqual(drain(machine), PHRASE + [CLOSE])
        self.assertTrue(machine.landed)
        return machine

    def test_flags_default_on_and_must_be_booleans(self):
        self.assertTrue(LandingConfig().thinking_triggers)
        self.assertTrue(LandingConfig().answer_guard)
        for name in ("thinking_triggers", "answer_guard"):
            with self.subTest(name=name), self.assertRaises(TypeError):
                LandingConfig(**{name: 1})

    def test_repeat_after_arming_does_not_trigger(self):
        machine = state(LandingConfig(thinking_triggers=False))
        self.prelude(machine)
        self.repeat(machine, self.SENTENCE, times=6)
        self.assertIsNone(machine.trigger)
        self.assertIsNone(machine.next_forced())

    def test_low_variety_loop_does_not_trigger(self):
        machine = state(LandingConfig(thinking_triggers=False, repeat_count=100000))
        self.prelude(machine)
        for i in range(1024):
            machine.observe(i % 4, " hmm")
        self.assertIsNone(machine.trigger)

    def test_budget_still_lands(self):
        machine = self.landed()
        self.assertEqual(machine.metadata()["trigger_thinking_count"], 5)
        self.assertTrue(machine.mask_open and machine.mask_close)

    def test_unguarded_answer_survives_repetition_and_old_cap(self):
        machine = self.landed(answer_guard=False)
        for _ in range(3000):
            machine.observe(7, " loop")
        self.assertFalse(machine.should_stop)
        self.assertIsNone(machine.answer_trigger)
        self.assertIsNone(machine.termination_reason)
        self.assertEqual(machine.answer_count, 3000)

    def test_guarded_answer_still_stops(self):
        machine = self.landed()
        for _ in range(3000):
            machine.observe(7, " loop")
        self.assertTrue(machine.should_stop)

    def test_request_can_reenable_either_flag(self):
        defaults = LandingConfig(thinking_triggers=False, answer_guard=False)
        result = LandingConfig.from_request(
            {"thinking_loop": {"thinking_triggers": True}}, defaults)
        self.assertTrue(result.thinking_triggers)
        self.assertFalse(result.answer_guard)

    def test_shipped_defaults_are_budget_only(self):
        path = Path(__file__).resolve().parents[1] / "serving/thinking_defaults.json"
        shipped = LandingConfig(**json.loads(path.read_text()))
        self.assertTrue(shipped.enabled)
        self.assertFalse(shipped.thinking_triggers)
        self.assertFalse(shipped.answer_guard)
        self.assertEqual(shipped.budget, 3000)
        self.assertEqual(shipped.bias_start_fraction, .5)
        self.assertEqual(shipped.collapse_max_distinct, 4)
        self.assertEqual(shipped.collapse_min_eligible, 64)
        self.assertEqual(shipped.collapse_ban_tokens, 256)


class CollapseTests(unittest.TestCase):
    """v56: the "Hmm, hmm." loop, with the checkpoint tokenizer's real IDs."""

    HMM, LOW, COMMA, STOP = 87459, 86776, 11, 13
    LOOP = [(87459, " Hmm"), (11, ","), (86776, " hmm"), (13, ".")]
    IRREGULAR = [(87459, " Hmm"), (13, "."), (87459, " Hmm"), (11, ","),
                 (86776, " hmm"), (13, "."), (87459, " Hmm"), (13, ".")]

    @staticmethod
    def config(**kwargs):
        return LandingConfig(**{"thinking_triggers": False, "answer_guard": False,
                                "collapse_max_distinct": 4, "grace_tokens": 0, **kwargs})

    @staticmethod
    def prelude(machine, count=300):
        for token in range(100000, 100000 + count):
            machine.observe(token, " unique")

    @staticmethod
    def loop(machine, pattern, count):
        for index in range(count):
            if machine.next_forced() is not None or machine.should_stop:
                return index
            machine.observe(*pattern[index % len(pattern)])
        return count

    def test_off_by_default_in_code(self):
        self.assertEqual(LandingConfig().collapse_max_distinct, 0)
        machine = state(self.config(collapse_max_distinct=0))
        self.prelude(machine)
        self.loop(machine, self.LOOP, 2000)
        self.assertIsNone(machine.trigger)

    def test_regular_and_irregular_loops_land_within_one_window(self):
        for name, pattern in (("regular", self.LOOP), ("irregular", self.IRREGULAR)):
            with self.subTest(pattern=name):
                machine = state(self.config())
                self.prelude(machine)
                used = self.loop(machine, pattern, 2000)
                self.assertEqual(machine.trigger, "collapse")
                self.assertGreater(used, 200)
                self.assertLessEqual(used, 256)
                self.assertEqual(drain(machine), PHRASE + [CLOSE])
                self.assertTrue(machine.landed)

    def test_no_arming_delay_and_ignores_soft_trigger_flags(self):
        machine = state(self.config())
        self.assertEqual(self.loop(machine, self.LOOP, 2000), 256)
        self.assertEqual(machine.trigger, "collapse")
        self.assertEqual(machine.metadata()["trigger_thinking_count"], 256)

    def test_ban_covers_loop_words_then_lapses(self):
        machine = state(self.config(collapse_ban_tokens=40))
        self.loop(machine, self.LOOP, 2000)
        self.assertEqual(machine.banned_ids, frozenset({self.HMM, self.LOW}))
        self.assertEqual(machine.metadata()["banned_ids"], [self.LOW, self.HMM])
        drain(machine)
        self.assertEqual(machine.banned_ids, frozenset({self.HMM, self.LOW}))
        for token in range(200000, 200040):
            machine.observe(token, " answer")
        self.assertEqual(machine.banned_ids, frozenset())
        self.assertEqual(machine.simulation().banned_ids, frozenset())

    def test_landed_answer_collapse_stops_generation(self):
        machine = state(self.config())
        self.loop(machine, self.LOOP, 2000)
        drain(machine)
        self.assertEqual(self.loop(machine, self.LOOP, 2000), 256)
        self.assertTrue(machine.should_stop)
        self.assertEqual(machine.termination_reason, "answer_collapse")
        self.assertEqual(machine.metadata()["answer_trigger_count"], 256)

    def test_natural_answer_is_not_guarded(self):
        machine = state(self.config())
        machine.observe(5, " fine")
        machine.observe(CLOSE, "</think>")
        self.assertFalse(machine.landed)
        self.loop(machine, self.LOOP, 2000)
        self.assertFalse(machine.should_stop)

    def test_formula_restating_with_many_symbols_is_not_collapse(self):
        machine = state(self.config())
        symbols = [(300 + i, " " + name) for i, name in enumerate(
            ["Theta", "Q", "L", "aE", "cot", "theta", "cos", "Sigma", "Delta", "r"])]
        pattern = [pair for symbol in symbols for pair in (symbol, (11, ","))]
        self.assertEqual(self.loop(machine, pattern, 3000), 3000)
        self.assertIsNone(machine.trigger)

    def test_too_few_word_tokens_is_not_collapse(self):
        machine = state(self.config())
        pattern = [(87459, " Hmm")] + [(20 + i % 10, str(i % 10)) for i in range(7)]
        self.assertEqual(self.loop(machine, pattern, 3000), 3000)
        self.assertIsNone(machine.trigger)

    def test_pure_arithmetic_is_not_collapse(self):
        machine = state(self.config())
        pattern = [(20, "0"), (11, ","), (21, "1"), (30, " ="), (31, " +")]
        self.assertEqual(self.loop(machine, pattern, 3000), 3000)
        self.assertIsNone(machine.trigger)

    def test_real_captured_reasoning_never_collapses(self):
        for filename in ("c_on_arithmetic.json", "v50_c_seed1.json", "v50_c_seed2.json"):
            fixture = json.loads((Path(__file__).parent / "fixtures" / filename)
                                 .read_text(encoding="utf-8"))
            with self.subTest(fixture=filename):
                machine = state(self.config())
                for token, piece in zip(fixture["sampled_thinking_ids"],
                                        fixture["sampled_thinking_pieces"], strict=True):
                    machine.observe(token, piece)
                self.assertIsNone(machine.trigger)

    def test_budget_still_wins_and_invalid_values_rejected(self):
        machine = state(self.config(budget=100))
        self.loop(machine, self.LOOP, 2000)
        self.assertEqual(machine.trigger, "budget")
        for values in (dict(collapse_max_distinct=-1), dict(collapse_max_distinct=True),
                       dict(collapse_min_eligible=0), dict(collapse_ban_tokens=1.5)):
            with self.subTest(values=values), self.assertRaises((ValueError, TypeError)):
                LandingConfig(**values)


if __name__ == "__main__":
    unittest.main()
