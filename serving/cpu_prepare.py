"""CPU-only renderer/control checks using the actual model tokenizer and template."""
import dataclasses,importlib.util,json,sys,unittest
from pathlib import Path
from types import SimpleNamespace as NS
sys.path.append('/opt/sglang/lib/python3.12/site-packages')
from tokenizers import Tokenizer
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.server.errors import RequestError
HERE=Path(__file__).resolve().parent;COPY=HERE.parent
spec=importlib.util.spec_from_file_location('copied_server_cpu',HERE/'server.py');server=importlib.util.module_from_spec(spec);spec.loader.exec_module(server)
MODEL=Path('/model');tok=Tokenizer.from_file(str(MODEL/'tokenizer.json'))
e=NS(w=NS(cfg=NS(eos=(154820,154827,154829))),st=NS(capacity=89344))
app=server.make_app((e,tok,None,None,Sampling,None,None,None,None),MODEL)
def body(**kw):return {'model':'glm53','messages':[{'role':'user','content':'Reply OK.'}],'max_tokens':16384,**kw}
class RequestControls(unittest.TestCase):
 def test_default_state_and_phase(self):
  p=app.prepare(body(),True);self.assertTrue(p.landing_state.thinking);self.assertEqual(p.landing_state.effective_budget,3000);self.assertEqual(p.think_budget,0)
 def test_missing_seed_is_fresh_and_explicit_seed_wins(self):
  a=app.prepare(body(),True).sampling.seed;b=app.prepare(body(),True).sampling.seed;self.assertNotEqual(a,b)
  self.assertEqual(app.prepare(body(seed=7),True).sampling.seed,7);self.assertIsNone(app.prepare(body(temperature=0),True).sampling)
 def test_prefix_cache_defaults_on_and_request_can_opt_out(self):
  self.assertTrue(app.prepare(body(),True).prefix_cache);self.assertFalse(app.prepare(body(prefix_cache=False),True).prefix_cache)
  with self.assertRaises(RequestError):app.prepare(body(prefix_cache=1),True)
  self.assertEqual((app.prefix_cache,app.keepalive_seconds),(True,15))
 def test_empty_budget(self):
  p=app.prepare(body(thinking_budget=0),True);self.assertEqual(p.landing_state.next_forced(),154842);self.assertEqual(p.landing_state.thinking_count,0)
 def test_nested_controls(self):
  p=app.prepare(body(thinking_loop={'budget':42,'repeat_count':4,'bias_max':6}),True);self.assertEqual(p.landing_state.config.budget,42);self.assertEqual(p.landing_state.config.repeat_count,4);self.assertEqual(p.landing_state.config.bias_max,6)
 def test_top_budget_wins_even_zero(self):
  p=app.prepare(body(thinking_budget=0,thinking_loop={'budget':42}),True);self.assertEqual(p.landing_state.next_forced(),154842)
 def test_feature_disabled_accepts_small_cap_and_original_path(self):
  p=app.prepare(body(max_tokens=128,thinking_loop={'enabled':False}),True);self.assertIsNone(p.landing_state);self.assertEqual(p.max_tokens,128)
 def test_invalid_controls_rejected(self):
  for controls in ({'answer_reserve':1023},{'grace_tokens':49},{'budget':False},{'bias_max':float('nan')},{'typo':True}):
   with self.subTest(controls=controls),self.assertRaises(RequestError):app.prepare(body(thinking_loop=controls),True)
 def test_small_cap_rejected_not_enlarged(self):
  with self.assertRaises(RequestError):app.prepare(body(max_tokens=1024),True)
 def test_budget_is_lowered_for_answer_and_overhead(self):
  p=app.prepare(body(max_tokens=1200),True);phrase=tok.encode(server.LANDING_TEXT,add_special_tokens=False).ids
  self.assertEqual(p.landing_state.effective_budget,1200-1024-len(phrase)-1-48);self.assertEqual(p.max_tokens,1200)
 def test_request_state_is_not_shared(self):
  one=app.prepare(body(),True);two=app.prepare(body(),True);one.landing_state.observe(8,'x');self.assertEqual(two.landing_state.thinking_count,0)
 def test_output_cap_not_raised(self):
  p=app.prepare(body(max_tokens=20000),True);self.assertEqual(p.max_tokens,16384)
if __name__=='__main__':
 print(json.dumps({'open_id':tok.token_to_id('<think>'),'close_id':tok.token_to_id('</think>'),'landing_ids':tok.encode(server.LANDING_TEXT,add_special_tokens=False).ids,'landing_text':server.LANDING_TEXT}),flush=True)
 unittest.main()
