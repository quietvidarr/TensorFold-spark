"""CPU stdlib contracts for the forced-answer lifecycle and speculation seam."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "src/tensorfold/engine/thinking_landing.py"
_SPEC = importlib.util.spec_from_file_location("_cpu_answer_landing", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
LandingConfig = _MODULE.LandingConfig
ThinkingLanding = _MODULE.ThinkingLanding

OPEN, CLOSE = 90001, 90002
PHRASE = [80001, 80002, 80003]
# Runtime-provided GLM tokenizer IDs. No tokenizer/model/GPU import is needed.
HEADER = [565, 21754, 271]
HEADER_PIECES = ["##", " Answer", "\n\n"]
SENTENCE = [" We", " should", " verify", " each", " assumption",
            " before", " giving", " answers."]


def state(config=None, **kwargs):
    return ThinkingLanding(config or LandingConfig(), OPEN, CLOSE, PHRASE,
                           kwargs.pop("output_cap", 12000),
                           answer_prefix_ids=kwargs.pop("answer_prefix_ids", HEADER), **kwargs)


def drain(machine):
    result = []
    while machine.next_forced() is not None and not machine.should_stop:
        token = machine.next_forced()
        result.append(token)
        machine.observe(token, "synthetic")
    return result


def landed(config=None, **kwargs):
    machine = state(config or LandingConfig(budget=0), **kwargs)
    if machine.next_forced() is None:
        raise AssertionError("helper needs an immediately queued landing")
    drain(machine)
    return machine


class AnswerConfigTests(unittest.TestCase):
    def test_cap_default_downward_override_and_inheritance(self):
        self.assertEqual(LandingConfig().answer_max_tokens, 2048)
        for limit in (1, 3, 1024, 2048):
            with self.subTest(limit=limit):
                config = LandingConfig.from_request({"thinking_loop": {"answer_max_tokens": limit}})
                self.assertEqual(config.answer_max_tokens, limit)
        config = LandingConfig(answer_max_tokens=37)
        self.assertEqual(LandingConfig.from_request({}, config), config)

    def test_cap_validation_constructor_and_request(self):
        for value in (True, False, 1.0, "2048", None, 0, -1, 2049):
            for factory in (lambda: LandingConfig(answer_max_tokens=value),
                            lambda: LandingConfig.from_request(
                                {"thinking_loop": {"answer_max_tokens": value}})):
                with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                    factory()

    def test_header_validation(self):
        for ids in ([OPEN], [CLOSE], [True], [-1], [1.0], ["565"]):
            with self.subTest(ids=ids), self.assertRaises((TypeError, ValueError)):
                state(answer_prefix_ids=ids)

    def test_header_is_keyword_only_and_default_is_compatible(self):
        machine = ThinkingLanding(LandingConfig(budget=0), OPEN, CLOSE, PHRASE, 12000)
        self.assertEqual(drain(machine), [CLOSE])
        self.assertEqual(machine.answer_count, 0)
        self.assertFalse(machine.serial_required)
        with self.assertRaises(TypeError):
            ThinkingLanding(LandingConfig(), OPEN, CLOSE, PHRASE, 12000, True, None, HEADER)


class AnswerLandingTests(unittest.TestCase):
    def test_forced_close_keeps_exact_header_queue_and_serial_seams(self):
        logs = []
        machine = state(LandingConfig(budget=1, grace_tokens=0), logger=logs.append)
        self.assertFalse(machine.serial_required)
        self.assertFalse(machine.mask_open)
        self.assertFalse(machine.mask_close)
        machine.observe(11, "done.")
        self.assertTrue(machine.serial_required)
        self.assertEqual(list(machine.forced), PHRASE + [CLOSE] + HEADER)
        for token in PHRASE:
            machine.observe(token, "forced phrase")
        self.assertFalse(machine.landed)
        self.assertEqual(machine.answer_count, 0)
        self.assertEqual(logs, [])
        machine.observe(CLOSE)
        self.assertTrue(machine.landed)
        self.assertTrue(machine.closed)
        self.assertFalse(machine.thinking)
        self.assertTrue(machine.mask_open)
        self.assertTrue(machine.mask_close)
        self.assertEqual(list(machine.forced), HEADER)
        self.assertEqual(machine.answer_count, 0)
        event = json.loads(logs[0])
        self.assertEqual(event["forced_remaining"], 3)
        self.assertTrue(event["serial_required"])
        self.assertEqual(event["answer_count"], 0)
        for count, (token, piece) in enumerate(zip(HEADER, HEADER_PIECES, strict=True), 1):
            self.assertTrue(machine.serial_required)
            machine.observe(token, piece)
            self.assertEqual(machine.answer_count, count)
            self.assertEqual(len(machine._ngram_tail), 0)
            self.assertEqual(len(machine._variety_tokens), 0)
        self.assertFalse(machine.serial_required)
        self.assertEqual(machine.next_forced(), None)
        self.assertEqual(machine.thinking_count, 4)
        self.assertEqual(machine.metadata()["forced_thinking_tokens"], 3)
        self.assertEqual(len(logs), 1)

    def test_serial_during_grace_and_until_empty_prefix(self):
        machine = state(LandingConfig(budget=1))
        machine.observe(11, "word")
        self.assertTrue(machine.serial_required)
        self.assertIsNone(machine.next_forced())
        machine.observe(12, "\n")
        self.assertEqual(drain(machine), PHRASE + [CLOSE] + HEADER)
        self.assertFalse(machine.serial_required)

    def test_zero_budget_forces_close_and_header_without_phrase(self):
        machine = state(LandingConfig(budget=0))
        self.assertEqual(list(machine.forced), [CLOSE] + HEADER)
        self.assertEqual(drain(machine), [CLOSE] + HEADER)
        self.assertEqual(machine.thinking_count, 0)
        self.assertEqual(machine.answer_count, 3)
        self.assertFalse(machine.should_stop)

    def test_natural_close_does_not_inject_header_or_arm_any_answer_guard(self):
        machine = state(LandingConfig(answer_max_tokens=1))
        machine.observe(CLOSE)
        self.assertEqual(list(machine.forced), [])
        self.assertFalse(machine.landed)
        self.assertTrue(machine.mask_open)
        self.assertTrue(machine.mask_close)
        for index in range(2100):
            machine.observe(index % 8, " prose")
        self.assertEqual(machine.answer_count, 2100)
        self.assertFalse(machine.should_stop)
        self.assertIsNone(machine.answer_trigger)
        self.assertIsNone(machine.termination_reason)

    def test_natural_close_cancels_phrase_and_header_including_during_grace(self):
        for grace in (0, 48):
            with self.subTest(grace=grace):
                machine = state(LandingConfig(budget=1, grace_tokens=grace,
                                              answer_max_tokens=1))
                machine.observe(11, "word")
                machine.observe(CLOSE)
                self.assertFalse(machine.landed)
                self.assertEqual(list(machine.forced), [])
                for token in range(100):
                    machine.observe(token, " answer")
                self.assertFalse(machine.should_stop)
                self.assertEqual(machine.answer_count, 100)

    def test_disabled_does_not_force_answer_or_limit_natural_answer(self):
        machine = state(LandingConfig(enabled=False, budget=0, answer_max_tokens=1), output_cap=1)
        self.assertIsNone(machine.next_forced())
        machine.observe(CLOSE)
        for index in range(300):
            machine.observe(index % 4, " word")
        self.assertFalse(machine.landed)
        self.assertFalse(machine.should_stop)

    def test_forced_phrase_and_header_never_feed_detectors(self):
        machine = ThinkingLanding(LandingConfig(budget=1, grace_tokens=0,
                                               repeat_ngram=1, repeat_count=2,
                                               variety_window=1, variety_fraction=1),
                                  OPEN, CLOSE, [10] * 8, 12000,
                                  answer_prefix_ids=[10] * 8)
        machine.observe(11, "word")
        self.assertEqual(drain(machine), [10] * 8 + [CLOSE] + [10] * 8)
        self.assertIsNone(machine.answer_trigger)
        self.assertEqual(machine.answer_count, 8)
        self.assertEqual(machine._ngram_counts, {})
        self.assertEqual(machine._variety_counts, {})
        machine.observe(10, "word")
        self.assertFalse(machine.should_stop)
        machine.observe(10, "word")
        self.assertEqual(machine.answer_trigger, "repeat_ngram")
        self.assertEqual(machine.answer_trigger_count, 10)

    def test_forced_close_resets_thinking_history_not_just_count(self):
        machine = state(LandingConfig(budget=16, grace_tokens=0))
        for _ in range(2):
            for token, piece in enumerate(SENTENCE):
                machine.observe(token, piece)
        self.assertEqual(machine.trigger, "budget")
        self.assertEqual(machine._ngram_counts[tuple(range(8))], 2)
        drain(machine)
        self.assertEqual(machine._ngram_counts, {})
        self.assertEqual(machine._variety_counts, {})
        self.assertEqual(machine.answer_count, 3)
        for cycle in range(3):
            for token, piece in enumerate(SENTENCE):
                machine.observe(token, piece)
            self.assertEqual(machine.should_stop, cycle == 2)
        self.assertEqual(machine.answer_trigger, "repeat_ngram")
        self.assertEqual(machine.answer_trigger_count, 27)
        self.assertEqual(machine.thinking_count, 19)


class AnswerDetectorTests(unittest.TestCase):
    def test_repeat_unguarded_by_1500_and_no_answer_budget(self):
        machine = landed()
        for _ in range(3):
            for token, piece in enumerate(SENTENCE):
                machine.observe(token, piece)
        self.assertEqual(machine.thinking_count, 0)
        self.assertEqual(machine.answer_count, 27)
        self.assertTrue(machine.should_stop)
        self.assertEqual(machine.answer_trigger, "repeat_ngram")
        self.assertEqual(machine.termination_reason, "answer_repeat_ngram")
        self.assertEqual(machine.trigger, "budget")
        self.assertEqual(machine.answer_trigger_count, 27)

    def test_low_variety_unguarded_by_1500_raw_window_excludes_header(self):
        machine = landed(LandingConfig(budget=0, repeat_count=100000))
        for index in range(255):
            machine.observe(index % 4, " word")
        self.assertFalse(machine.should_stop)
        machine.observe(3, " word")
        self.assertTrue(machine.should_stop)
        self.assertEqual(machine.answer_trigger, "low_variety")
        self.assertEqual(machine.termination_reason, "answer_low_variety")
        self.assertEqual(machine.answer_trigger_count, 259)

    def test_arithmetic_filter_applies_to_both_answer_detectors(self):
        machine = landed()
        for index in range(1800):
            machine.observe(index % 8, str(index % 8))
        self.assertIsNone(machine.answer_trigger)
        self.assertEqual(machine._variety_eligible, 0)
        self.assertEqual(machine.answer_count, 1803)

    def test_answer_mixed_numeric_denominator_and_exact_ratio(self):
        machine = landed(LandingConfig(budget=0, repeat_count=100000))
        for index in range(248):
            machine.observe(99, "123")
        for index in range(8):
            machine.observe(index % 2, " word")
        self.assertFalse(machine.should_stop)
        self.assertEqual(machine._variety_eligible, 8)
        machine.observe(0, " word")
        self.assertEqual(machine.answer_trigger, "low_variety")
        self.assertEqual(machine.answer_trigger_count, 260)

    def test_repeat_exact_512_token_eviction_in_answer_channel(self):
        for gap, expected in [(488, "repeat_ngram"), (489, None)]:
            with self.subTest(gap=gap):
                machine = landed()
                for token, piece in enumerate(SENTENCE):
                    machine.observe(token, piece)
                for token in range(100000, 100000 + gap):
                    machine.observe(token, " unique")
                for _ in range(2):
                    for token, piece in enumerate(SENTENCE):
                        machine.observe(token, piece)
                self.assertEqual(machine.answer_trigger, expected)

    def test_default_exact_2048_cap_includes_prefix(self):
        machine = landed()
        for token in range(100000, 100000 + 2044):
            machine.observe(token, " unique")
        self.assertEqual(machine.answer_count, 2047)
        self.assertFalse(machine.should_stop)
        machine.observe(200000, " unique")
        self.assertEqual(machine.answer_count, 2048)
        self.assertTrue(machine.should_stop)
        self.assertEqual(machine.answer_trigger, "answer_cap")
        self.assertEqual(machine.answer_trigger_count, 2048)
        self.assertEqual(machine.termination_reason, "answer_cap")
        self.assertFalse(machine.serial_required)

    def test_downward_cap_including_prefix_and_exact_small_boundaries(self):
        for limit in (1, 2, 3, 4, 1024):
            with self.subTest(limit=limit):
                machine = landed(LandingConfig(budget=0, answer_max_tokens=limit))
                while not machine.should_stop:
                    machine.observe(100000 + machine.answer_count, " unique")
                self.assertEqual(machine.answer_count, limit)
                self.assertEqual(machine.answer_trigger_count, limit)
                self.assertEqual(machine.answer_trigger, "answer_cap")
                if limit < 3:
                    self.assertEqual(list(machine.forced), HEADER[limit:])

    def test_cap_wins_when_repeat_or_variety_fire_on_same_token(self):
        for config, tokens in [
            (LandingConfig(budget=0, answer_max_tokens=27), list(range(8)) * 3),
            (LandingConfig(budget=0, answer_max_tokens=259, repeat_count=100000),
             [index % 4 for index in range(256)])
        ]:
            with self.subTest(config=config):
                machine = landed(config)
                for token in tokens:
                    machine.observe(token, " word")
                self.assertEqual(machine.answer_trigger, "answer_cap")

    def test_metadata_and_answer_termination_log_capture_first_stop(self):
        logs = []
        machine = landed(LandingConfig(budget=0, answer_max_tokens=4), logger=logs.append)
        machine.observe(15, " answer")
        metadata = machine.metadata()
        self.assertEqual(metadata["answer_count"], 4)
        self.assertEqual(metadata["answer_trigger"], "answer_cap")
        self.assertEqual(metadata["answer_trigger_count"], 4)
        self.assertEqual(metadata["termination_reason"], "answer_cap")
        self.assertEqual(metadata["answer_max_tokens"], 4)
        self.assertTrue(metadata["mask_open"])
        self.assertTrue(metadata["mask_close"])
        self.assertTrue(metadata["should_stop"])
        self.assertFalse(metadata["serial_required"])
        self.assertEqual(len(logs), 2)
        event = json.loads(logs[1])
        self.assertEqual(event["event"], "answer_termination")
        self.assertEqual(event["termination_reason"], "answer_cap")
        self.assertEqual(event["answer_count"], 4)
        machine.observe(16, " an incorrectly committed extra token")
        self.assertEqual(machine.answer_trigger_count, 4)
        self.assertEqual(len(logs), 2)


class CapacityAndSimulationTests(unittest.TestCase):
    def test_prefix_length_reduces_effective_budget_and_is_reserved(self):
        machine = state(output_cap=1100)
        self.assertEqual(machine.effective_budget, 21)
        for token in range(machine.effective_budget + 48):
            machine.observe(token, "word")
        self.assertEqual(drain(machine), PHRASE + [CLOSE] + HEADER)
        self.assertEqual(machine.output_cap - machine.total_emitted, 1024)
        self.assertEqual(machine.answer_count, 3)

    def test_prior_prefix_open_and_maximum_grace_preserve_header_room(self):
        machine = state(output_cap=1150, initial_thinking=False)
        for token in range(25):
            machine.observe(token)
        machine.observe(OPEN)
        self.assertEqual(machine.effective_budget, 45)
        for token in range(machine.effective_budget + 48):
            machine.observe(token, "word")
        drain(machine)
        self.assertEqual(machine.output_cap - machine.total_emitted, 1024)
        self.assertEqual(machine.answer_count, 3)

    def test_exact_minimum_capacity_and_clamped_zero_with_header(self):
        with self.assertRaises(ValueError):
            state(output_cap=1030)
        machine = state(output_cap=1031)
        self.assertEqual(machine.effective_budget, 0)
        self.assertEqual(drain(machine), PHRASE + [CLOSE] + HEADER)
        self.assertEqual(machine.output_cap - machine.total_emitted, 1024)
        with self.assertRaises(ValueError):
            state(LandingConfig(budget=0), output_cap=1027)
        machine = state(LandingConfig(budget=0), output_cap=1028)
        self.assertEqual(drain(machine), [CLOSE] + HEADER)
        self.assertEqual(machine.output_cap - machine.total_emitted, 1024)

    def test_simulation_is_deep_independent_and_logs_only_real_commit(self):
        logs = []
        machine = state(LandingConfig(budget=1, grace_tokens=0), logger=logs.append)
        before = machine.metadata()
        clone = machine.simulation()
        self.assertIsNone(clone.logger)
        clone.observe(10, "done.")
        self.assertEqual(drain(clone), PHRASE + [CLOSE] + HEADER)
        self.assertEqual(machine.metadata(), before)
        self.assertEqual(logs, [])
        self.assertIsNot(clone.forced, machine.forced)
        self.assertIsNot(clone._ngram_counts, machine._ngram_counts)
        machine.observe(10, "done.")
        drain(machine)
        self.assertEqual(machine.metadata(), clone.metadata())
        self.assertEqual(len(logs), 1)

    def test_simulation_can_advance_pending_header_without_mutating_original(self):
        logs = []
        machine = state(LandingConfig(budget=0), logger=logs.append)
        machine.observe(CLOSE)
        before = machine.metadata()
        clone = machine.simulation()
        self.assertEqual(drain(clone), HEADER)
        self.assertFalse(clone.serial_required)
        self.assertEqual(machine.metadata(), before)
        self.assertEqual(list(machine.forced), HEADER)
        self.assertEqual(len(logs), 1)
        drain(machine)
        self.assertEqual(machine.metadata(), clone.metadata())

    def test_simulation_repeat_and_cap_stop_do_not_emit_log_or_touch_histories(self):
        for cap in (2048, 27):
            with self.subTest(cap=cap):
                logs = []
                machine = landed(LandingConfig(budget=0, answer_max_tokens=cap), logger=logs.append)
                for token in list(range(8)) * 2:
                    machine.observe(token, " prose")
                before = machine.metadata()
                counts = machine._ngram_counts.copy()
                clone = machine.simulation()
                for token in range(8):
                    clone.observe(token, " prose")
                self.assertTrue(clone.should_stop)
                self.assertFalse(machine.should_stop)
                self.assertEqual(machine.metadata(), before)
                self.assertEqual(machine._ngram_counts, counts)
                self.assertEqual(len(logs), 1)
                for token in range(8):
                    machine.observe(token, " prose")
                self.assertEqual(machine.metadata(), clone.metadata())
                self.assertEqual(len(logs), 2)

    def test_simulation_does_not_deepcopy_uncloneable_logger(self):
        class Logger:
            def __deepcopy__(self, memo):
                raise AssertionError("logger must not be copied")

            def info(self, message):
                raise AssertionError("simulation must not log")

        machine = state(LandingConfig(budget=0), logger=Logger())
        clone = machine.simulation()
        drain(clone)
        self.assertTrue(clone.landed)
        self.assertFalse(machine.closed)


if __name__ == "__main__":
    unittest.main()
