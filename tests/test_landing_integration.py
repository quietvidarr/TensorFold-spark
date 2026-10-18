"""CPU hook/adapter tests. Synthetic fixtures, not model outputs or qualification."""
import ast,dataclasses,importlib.util,sys,unittest
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from tensorfold.engine.thinking_landing import LandingConfig,ThinkingLanding
COPY=Path(__file__).resolve().parents[1]
source=COPY/'src/tensorfold/families/glm5_next/cuda/decode.py'
tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='Engine');fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='sample')
unit=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[]);ast.fix_missing_locations(unit)
def sampler(w,logits,positions,sampling,offset,probs):return logits.argmax(dim=-1).tolist()
ns={'sample_rows':sampler};exec(compile(unit,str(source),'exec'),ns);sample_hook=ns['sample']
spec=importlib.util.spec_from_file_location('copied_serving',COPY/'serving/server.py');serving=importlib.util.module_from_spec(spec);spec.loader.exec_module(serving)

def state(**kw):return ThinkingLanding(LandingConfig(**kw),1,2,[3,4],2000)
def engine(landing):return NS(w=NS(cfg=NS(eos=(9,)),vocab_offset=0),constraint=None,window=None,thinking_landing=landing,last_hidden=None)
class Hooks(unittest.TestCase):
 def test_open_and_eos_masked_inside_thinking_without_buffer_mutation(self):
  e=engine(state());logits=torch.zeros(1,16);logits[0,1]=100;logits[0,9]=50;original=logits.clone()
  self.assertNotIn(sample_hook(e,logits,[0],None)[0],(1,9));self.assertTrue(torch.equal(logits,original))
 def test_after_close_masks_open_but_allows_eos(self):
  s=state();s.observe(2,'</think>');e=engine(s);logits=torch.zeros(1,16);logits[0,1]=100;logits[0,9]=50
  self.assertEqual(sample_hook(e,logits,[0],None),[9])
 def test_disabled_is_unmodified(self):
  e=engine(state(enabled=False));logits=torch.zeros(1,16);logits[0,1]=100
  self.assertEqual(sample_hook(e,logits,[0],None),[1])
 def test_collapse_ban_masks_loop_tokens_until_it_lapses(self):
  s=state(thinking_triggers=False,collapse_max_distinct=4,variety_window=8,collapse_min_eligible=2,collapse_ban_tokens=6,grace_tokens=0);e=engine(s)
  for i in range(8):s.observe(5+i%2,'hmm')
  self.assertEqual(s.trigger,'collapse');self.assertEqual(s.banned_ids,frozenset({5,6}))
  logits=torch.zeros(1,16);logits[0,5]=100;logits[0,6]=90;logits[0,7]=1;original=logits.clone()
  self.assertEqual(sample_hook(e,logits,[0],None,draft=True),[7]);self.assertTrue(torch.equal(logits,original))
  for t in (3,4,2):self.assertEqual(sample_hook(e,logits,[0],None),[t]);s.observe(t,'x')
  self.assertTrue(s.landed);self.assertEqual(sample_hook(e,logits,[0],None),[7])
  for t in (7,7,7):s.observe(t,'x')
  self.assertEqual(s.banned_ids,frozenset());self.assertEqual(sample_hook(e,logits,[0],None),[5])
 def test_forced_token_is_not_consumed_until_emitted(self):
  s=state(budget=0);e=engine(s);logits=torch.zeros(1,16)
  self.assertEqual(sample_hook(e,logits,[0],None),[2]);self.assertEqual(s.next_forced(),2);s.observe(2,'</think>');self.assertIsNone(s.next_forced())
 def test_close_bias_influences_real_sampler(self):
  s=state(budget=6,grace_tokens=1);e=engine(s)
  for t in (5,6,7,8):s.observe(t,'x')
  logits=torch.zeros(1,16);logits[0,5]=1
  self.assertEqual(sample_hook(e,logits,[0],None),[2]);self.assertEqual(logits[0,2].item(),0)
 def test_initial_phase_ignores_old_messages(self):
  tok=NS(token_to_id=lambda t:{'<|assistant|>':10,'<think>':1,'</think>':2}[t])
  self.assertFalse(serving.initial_thinking(tok,[10,1,7,2,10,8]))
  self.assertTrue(serving.initial_thinking(tok,[10,1,7,2,10,1]))
 def test_forced_tokens_are_assimilated_before_answer(self):
  s=ThinkingLanding(LandingConfig(budget=2,grace_tokens=1),1,2,[3,4],2000,answer_prefix_ids=[11,12,13]);cfg=NS(eos=(9,));e=engine(s);e.st=NS(pos=0,capacity=2000);e.buf=None;e.graphs=None;e.replays={};e.context=[];e.history=[]
  def reset():e.st.pos=0;e.context=[];e.history=[]
  def logits():
   x=torch.zeros(1,16)
   chosen=(9 if 8 in e.history else 8)if s.closed else 5+len(e.history)%3
   x[0,chosen]=10;return x
  def forward(tokens):e.context.extend(tokens);return logits()
  e.reset=reset;e.forward=forward;e.verify_window=lambda ts:ts;e.sample=lambda x,p,a:sample_hook(e,x,p,a);e.follow=lambda ts:e.history.extend(ts)
  def prefill(e,prompt,sampling,**kw):e.reset();e.context=list(prompt);e.st.pos=len(prompt);first=e.sample(logits(),[len(prompt)],sampling)[0];e.follow([first]);return first
  def commit(w,st,buf,rows,keep):st.pos+=keep
  api=NS(prefill=prefill,commit=commit,take_snapshot=lambda e,ids,pending,**kw:NS(ids=list(ids),mtp_len=-1));tok=NS(decode=lambda ids,**kw:''.join({7:'\n',2:'</think>'}.get(i,'x')for i in ids))
  cuda=NS(synchronize=lambda:None,max_memory_reserved=lambda:0);fake_torch=NS(inference_mode=torch.inference_mode,cuda=cuda);hooks=NS(guard=lambda phase:None);sampling=dataclasses.make_dataclass('Sampling',[('temperature',float)])(1.0)
  runtime=(e,tok,fake_torch,api,object,None,None,hooks,{})
  e.thinking_landing=None
  out=[];stats=serving.generate(runtime,[10,1],2000,sampling,lambda ts:out.extend(ts)or False,draft=False,landing_state=s)
  self.assertEqual(out,[5,6,7,3,4,2,11,12,13,8,9]);self.assertEqual(e.context,[10,1]+out[:-1]);self.assertTrue(stats['thinking_loop']['landed'])
  self.assertIsNone(e.thinking_landing) # no request-state leak

if __name__=='__main__':unittest.main()
