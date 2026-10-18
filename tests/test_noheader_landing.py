"""CPU state and exact MTP seam checks for an empty forced-answer prefix."""
import json,unittest
from pathlib import Path
from tests.test_landing_mtp import Fake,TrackedLanding,LandingConfig
ROOT=Path(__file__).resolve().parents[3]
class NoHeader(unittest.TestCase):
 def state(self,**kw):
  return TrackedLanding(LandingConfig(**{'budget':2,'grace_tokens':0,'bias_max':0,**kw}),1,2,[3,4],4096,answer_prefix_ids=())
 def test_v55_defaults_are_v51_plus_budget_only_changes(self):
  here=Path(__file__).resolve().parents[1]
  v51=json.loads((ROOT/'thinking-v51/TensorFold/serving/thinking_defaults.json').read_text())
  self.assertEqual(json.loads((here/'serving/thinking_defaults.json').read_text()),{**v51,'budget':3000,'thinking_triggers':False,'answer_guard':False,'collapse_max_distinct':4,'collapse_min_eligible':64,'collapse_ban_tokens':256})
 def test_queue_is_phrase_then_close_only(self):
  s=self.state();s.observe(5,'word');s.observe(6,'word')
  self.assertEqual(list(s.forced),[3,4,2])
  for token in (3,4,2):s.observe(token,'word')
  self.assertFalse(s.serial_required);self.assertEqual(s.answer_count,0);self.assertEqual(s.next_forced(),None)
  self.assertTrue(s.mask_open);self.assertTrue(s.mask_close)
 def test_mtp_resumes_immediately_after_close(self):
  s=self.state();f=Fake([5,6,7,8,7,8,9],s);stats=f.run(4096)
  self.assertEqual(f.out,[5,6,3,4,2,8,9]);self.assertEqual(s.observed,f.out)
  self.assertEqual(stats['mtp_resumes'],1);self.assertGreater(stats['serial_rounds'],0);self.assertGreater(stats['drafted'],0)
  self.assertEqual(s.answer_count,2);self.assertFalse(s.serial_required)
 def test_answer_cap_counts_only_sampled_answer(self):
  s=self.state(answer_max_tokens=3);f=Fake([5,6,7,8,7]+[8]*10,s);f.run(4096)
  self.assertEqual(f.out,[5,6,3,4,2,8,8,8]);self.assertEqual(s.answer_count,3);self.assertEqual(s.termination_reason,'answer_cap')
 def test_zero_budget_forces_only_close(self):
  s=self.state(budget=0);self.assertEqual(list(s.forced),[2]);s.observe(2)
  self.assertEqual(s.answer_count,0);self.assertFalse(s.serial_required)
 def test_natural_close_has_no_prefix(self):
  s=self.state(budget=100);f=Fake([5,2,8,9],s);f.run(4096)
  self.assertEqual(f.out,[5,2,8,9]);self.assertFalse(s.landed);self.assertEqual(s.answer_prefix_ids,())
if __name__=='__main__':unittest.main()
