"""Owned guarded HTTP adapter over the exercised v43 TensorFold sparse runtime.
No model/source edits. Tools and native vision bridge, live prefix resume, exact cap tails.
"""
import argparse,collections,dataclasses,importlib.util,json,os,secrets,sys,time
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent;COPY=HERE.parent;ROOT=HERE.parents[2];STUDY=COPY.parent;STAGE='tfglm-thinking-v59-pipeline';MAX_OUTPUT=16384
# context (since v58): 262,144 cache slots. The prompt ceiling leaves 4,096 tokens of reply inside the window.
SLOTS=262144;CONTEXT=SLOTS-128;MAX_PROMPT=CONTEXT-4096
# Host memory this copy keeps free. The inherited v42 guard and loader keep their own 20 GiB figure.
RESERVE_GIB=15
# v42 measured 87,306,534,912 bytes reserved at 89,344 slots. The source geometry puts 262,144 slots
# 3.42 GiB above that; 3.5 GiB is added here. An estimate until this copy's own peak is measured.
PEAK_ESTIMATE=87306534912+3758096384
LANDING_TEXT="I'll stop deliberating here, go with the most likely form, and proceed. Anything unverified I'll check with tools or flag."

def import_path(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

class PrefixCache:
 """Prompt states still true in the one engine's live caches: the state a prefill resumed from and the one it ended in.
 A prefill rewrites every row from its resume point, so every other kept state is dropped before it starts."""
 def __init__(self):self.snaps=[]
 def find(self,prompt,mtp):
  best=None
  for s in self.snaps:
   n=len(s.ids)
   if n<len(prompt) and (not mtp or s.mtp_len>=0) and prompt[:n]==s.ids and (best is None or n>len(best.ids)):best=s
  return best
 def begin(self,hit):self.snaps=[hit] if hit is not None else []
 def keep(self,snap):self.snaps=[s for s in self.snaps if s.ids!=snap.ids]+[snap]

def prefix_states(e):
 # The kept states describe one engine's caches, so they live on that engine.
 if getattr(e,'prefix_states',None) is None:e.prefix_states=PrefixCache()
 return e.prefix_states

def cached_prefill(runtime,prompt,sampling,image,use_cache,cancelled):
 """Prefill from the longest live strict prefix, else from an engine reset. Returns the first token and the resumed length."""
 e,api=runtime[0],runtime[3];mtp=not image;prompt=list(prompt);cache=prefix_states(e)
 hit=cache.find(prompt,mtp) if use_cache and not image else None
 cache.begin(hit)
 if hit is None:e.reset()
 try:first=api.prefill(e,prompt,sampling,mtp=mtp,resume=hit,stop=cancelled)
 except BaseException:
  cache.begin(None);raise  # rows past the resume point are half written: the next prompt starts from a reset
 if not image:cache.keep(api.take_snapshot(e,prompt,e.last_hidden if mtp else None,mtp=mtp))
 return first,(len(hit.ids) if hit is not None else 0)

def _generate_unmodified(runtime,prompt,count,sampling,on_tokens,draft=True,stop_eos=True,image=False,prefix_cache=True,cancelled=None):
 e,tok,torch,api,Sampling,scratch,forward,hooks,admission=runtime
 guard=hooks.guard;out=[];rounds=proposals=accepted=0;start=time.perf_counter();stamps=[]
 with torch.inference_mode():
  guard('http_request_start');torch.cuda.synchronize()
  first,cached=cached_prefill(runtime,prompt,sampling,image,prefix_cache,cancelled);out.append(first);torch.cuda.synchronize();stamps.append(time.perf_counter())
  stop=bool(on_tokens([first]));pending=[first];hidden=scratch[:1];hidden.copy_(e.last_hidden)
  while len(out)<count and not stop and not(stop_eos and out[-1] in e.w.cfg.eos):
   guard('http_each_decode');room=e.st.capacity-e.st.pos
   if room<1:raise ValueError('No cache headroom')
   use=draft and not image and count-len(out)>=2 and room>=2 and e.st.mtp_len+len(pending)<=e.st.capacity
   ds=api.draft(e,hidden,pending,e.st.pos+1,1,sampling) if use else []
   ts=e.verify_window([out[-1]]+ds);ds=ts[1:];R=len(ts);y=e.forward(ts);torch.cuda.synchronize()
   sampled=e.sample(y[:R],[e.st.pos+1+r for r in range(R)],sampling)
   keep=1+int(bool(ds) and sampled[0]==ds[0] and not(stop_eos and sampled[0] in e.w.cfg.eos))
   pos=e.st.pos;api.commit(e.w,e.st,e.buf,R,keep)
   if e.st.pos!=pos+keep:raise ValueError('Commit advance mismatch')
   rounds+=1;proposals+=len(ds);accepted+=keep-1
   pending=sampled[:keep];e.follow(pending);out.extend(pending);torch.cuda.synchronize();stamps.append(time.perf_counter())
   stop=bool(on_tokens(pending));hidden=scratch[:keep];hidden.copy_(e.main_hidden(slice(0,keep)))
  guard('http_request_finished')
 if len(out)>count:raise ValueError('Output cap overshoot')
 ttft=stamps[0]-start;decode=stamps[-1]-stamps[0]
 return dict(cached=cached,prefill_s=ttft,decode_s=decode,decode_tok_s=(len(out)-1)/decode if decode else None,rounds=rounds,drafted=proposals,accepted=accepted,mtp_n=0 if image else 1,graphs=e.graphs is not None,vision=image,cache_slots=e.st.capacity,sparse_graph_replays=dict(e.replays),peak_reserved=torch.cuda.max_memory_reserved(),sampling=dataclasses.asdict(sampling) if sampling is not None else None)

def initial_thinking(tok,prompt):
 # Only the final assistant suffix, not reasoning in old messages.
 assistant=tok.token_to_id('<|assistant|>');open_id=tok.token_to_id('<think>');close_id=tok.token_to_id('</think>')
 start=max((i for i,t in enumerate(prompt) if t==assistant),default=-1)
 thinking=False
 for t in prompt[start+1:]:
  if t==open_id:thinking=True
  elif t==close_id:thinking=False
 return thinking

def generate(runtime,prompt,count,sampling,on_tokens,draft=True,stop_eos=True,image=False,landing_state=None,constraint=None,prefix_cache=True,cancelled=None):
 # Opt-in only: ordinary chat takes exactly the inherited v53 path. The same
 # target masks prune speculative windows; MTP proposals stay unconstrained.
 if constraint is None:
  return _generate(runtime,prompt,count,sampling,on_tokens,draft,stop_eos,image,landing_state,prefix_cache,cancelled)
 if image or not stop_eos:raise ValueError('JSON constraints require text-only generation with EOS enabled')
 if landing_state is not None and landing_state.answer_prefix_ids:
  raise ValueError('JSON constraints cannot be combined with a forced answer prefix')
 e=runtime[0];old_constraint=e.constraint;old_window=e.window
 e.constraint=constraint;e.window=None
 try:
  stats=_generate(runtime,prompt,count,sampling,on_tokens,draft,stop_eos,image,landing_state,prefix_cache,cancelled)
  stats['structured_output']=dict(type='json_object',grammar_terminated=constraint.finished,
                                 reasoning_closed=constraint.active,mtp_preserved=bool(draft))
  return stats
 finally:
  e.constraint=old_constraint;e.window=old_window

def _generate(runtime,prompt,count,sampling,on_tokens,draft=True,stop_eos=True,image=False,landing_state=None,prefix_cache=True,cancelled=None):
 if landing_state is None or not landing_state.config.enabled:
  return _generate_unmodified(runtime,prompt,count,sampling,on_tokens,draft,stop_eos,image,prefix_cache,cancelled)
 # Target rows are sampled individually. A logger-free simulation advances only
 # over the guaranteed accepted first candidate, never over proposal tokens.
 e,tok,torch,api,Sampling,scratch,forward,hooks,admission=runtime
 guard=hooks.guard;out=[];stamps=[];rounds=proposals=accepted=serial_rounds=mtp_resumes=0
 start=time.perf_counter();old=getattr(e,'thinking_landing',None);old_window=getattr(e,'window',None)
 e.thinking_landing=landing_state
 mtp_enabled=draft and not image
 def requires_serial(state):
  # A natural close cancels a triggered grace/landing phase. A historical
  # trigger alone must not suppress MTP forever after its queue is gone.
  return state.serial_required and (not state.closed or state.next_forced() is not None)
 was_serial=requires_serial(landing_state)
 def observe(state,token):
  state.observe(int(token),tok.decode([int(token)],skip_special_tokens=False))
 def emit(tokens):
  # A whole accepted window is committed before callbacks. Even cancellation
  # on its first streamed piece must not hide any accepted token from the state.
  for token in tokens:observe(landing_state,token)
  out.extend(int(token) for token in tokens)
  torch.cuda.synchronize();stamps.append(time.perf_counter())
  return bool(on_tokens(list(tokens)))
 try:
  with torch.inference_mode():
   guard('http_request_start');torch.cuda.synchronize()
   first,cached=cached_prefill(runtime,prompt,sampling,image,prefix_cache,cancelled)
   stop=emit([first]);pending=[first]
   hidden=scratch[:1].copy_(e.last_hidden) if mtp_enabled else None
   while len(out)<count and not stop and not landing_state.should_stop and not(stop_eos and out[-1] in e.w.cfg.eos):
    guard('http_each_decode');room=e.st.capacity-e.st.pos
    if room<1:raise ValueError('No cache headroom')
    serial=requires_serial(landing_state)
    if serial:was_serial=True
    use=mtp_enabled and not serial and count-len(out)>=2 and room>=2 and e.st.mtp_len+len(pending)<=e.st.capacity
    if use and was_serial:mtp_resumes+=1;was_serial=False
    if mtp_enabled:
     # Actual api.draft(count=0) calls absorb(): it rolls back mtp_drafted
     # then assimilates the previous committed hidden/next-token pairs, without
     # sampling any proposal. Keeping this synchronized during the serial phase
     # reseeds every forced phrase/close/header row before MTP resumes.
     if e.st.mtp_len-e.st.mtp_drafted+len(pending)>e.st.capacity:raise ValueError('No MTP cache headroom')
     ds=api.draft(e,hidden,pending,e.st.pos+1,1 if use else 0,sampling)
     if e.st.mtp_len-e.st.mtp_drafted!=e.st.pos:raise ValueError('MTP committed-prefix alignment mismatch')
    else:ds=[]
    proposal_count=len(ds)
    ts=e.verify_window([out[-1]]+ds);ds=ts[1:];R=len(ts)
    y=e.forward(ts);torch.cuda.synchronize()
    # The base grammar's speculative window is target-only. Mask its full
    # window once before sampling separate rows; never mutate graph outputs.
    if e.constraint is not None and e.window is not None:
     y=y.float().clone();e.constraint.mask(y,e.window,e.w.vocab_offset);e.window=None
    first=e.sample(y[:1],[e.st.pos+1],sampling)[0];sampled=[first];keep=1
    if ds and first==ds[0] and not(stop_eos and first in e.w.cfg.eos):
     snapshot=landing_state.simulation();observe(snapshot,first)
     boundary=(snapshot.closed!=landing_state.closed or snapshot.trigger!=landing_state.trigger)
     if not boundary and not requires_serial(snapshot) and not snapshot.should_stop and count-len(out)>=2:
      try:
       e.thinking_landing=snapshot
       sampled.extend(e.sample(y[1:2],[e.st.pos+2],sampling));keep=2
      finally:e.thinking_landing=landing_state
    pos=e.st.pos;api.commit(e.w,e.st,e.buf,R,keep)
    if e.st.pos!=pos+keep:raise ValueError('Commit advance mismatch')
    rounds+=1;proposals+=proposal_count;accepted+=keep-1;serial_rounds+=int(not proposal_count)
    pending=sampled[:keep];e.follow(pending);stop=emit(pending)
    if mtp_enabled:hidden=scratch[:keep].copy_(e.main_hidden(slice(0,keep)))
   guard('http_request_finished')
 finally:
  e.thinking_landing=old;e.window=old_window
 if len(out)>count:raise ValueError('Output cap overshoot')
 ttft=stamps[0]-start;decode=stamps[-1]-stamps[0]
 return dict(cached=cached,prefill_s=ttft,decode_s=decode,decode_tok_s=(len(out)-1)/decode if decode else None,rounds=rounds,drafted=proposals,accepted=accepted,serial_rounds=serial_rounds,mtp_resumes=mtp_resumes,mtp_n=int(mtp_enabled),graphs=e.graphs is not None,vision=image,cache_slots=e.st.capacity,sparse_graph_replays=dict(e.replays),peak_reserved=torch.cuda.max_memory_reserved(),sampling=dataclasses.asdict(sampling) if sampling is not None else None,thinking_loop=landing_state.metadata(),stop_reason=landing_state.termination_reason,thinking_decode='mtp-target-with-protected-serial-landing')

def make_app(runtime,model,frontend=None):
 from tensorfold.cuda.server import App,RequestError
 e,tok,torch,api,Sampling,scratch,forward,hooks,admission=runtime
 class Adapter:
  eos=e.w.cfg.eos;tp=1;limit=CONTEXT;context_window=CONTEXT
  def generate(self,prompt,count,sampling,on_tokens,*,draft=True,stop_eos=True,vision=None,landing_state=None,constraint=None,prefix_cache=True,cancelled=None):
   if vision is None:return generate(runtime,prompt,count,sampling,on_tokens,draft,stop_eos,landing_state=landing_state,constraint=constraint,prefix_cache=prefix_cache,cancelled=cancelled)
   if constraint is not None:raise RequestError('JSON constraints with vision are not qualified')
   # An image prompt is never resumed or kept: its rows are not a text prefix.
   with frontend.context(vision,prompt):return generate(runtime,prompt,count,sampling,on_tokens,False,stop_eos,image=True,landing_state=landing_state,prefix_cache=False,cancelled=cancelled)
 class ServingApp(App):
  def _requested_tokens(self,body):
   # Bound larger client defaults to the user-selected output limit.
   return min(super()._requested_tokens(body),MAX_OUTPUT)
  def sampling_for(self,body,prompt):
   # Upstream derives a missing seed from the prompt, so a client retry replays
   # the same reply, loop included. Draw a fresh one; an explicit seed still wins.
   if body.get('seed') is None:body=dict(body,seed=secrets.randbits(62))
   return super().sampling_for(body,prompt)
  def _check_fields(self,body):
   error=super()._check_fields(body)
   if error:return error
   if body.get('model') not in (None,'glm53','glm53-tensorfold'):return 'Unknown model; use glm53'
   if body.get('reasoning_budget_tokens') is not None:return 'Use thinking_budget, not reasoning_budget_tokens'
   if any(body.get(k) is not None for k in ['guided_json','guided_regex','guided_choice','guided_grammar','structured_outputs']):return 'Use response_format type json_object; other constraint interfaces are unsupported'
   rf=body.get('response_format')
   if rf is not None:
    if rf.get('type') not in ('text','json_object'):return 'Only response_format type text or json_object is qualified; json_schema is unsupported'
    if set(rf)!={'type'}:return 'response_format only accepts the type field on this lane'
    if rf['type']=='json_object':
     from tensorfold.server.prompts import has_images
     if body.get('tools') or body.get('tool_choice') is not None:return 'json_object cannot be combined with tools or tool_choice'
     if has_images(body.get('messages',[])):return 'json_object with vision is not qualified'
     if body.get('ignore_eos') or body.get('stop'):return 'json_object requires EOS enabled and no stop strings'
   if body.get('n',1)!=1:return 'One completion per request only'
   return None
  def _prepare(self,body,chat):
   from tensorfold.engine.thinking_landing import LandingConfig,ThinkingLanding
   from tensorfold.server.prompts import has_images
   try:config=LandingConfig.from_request(body,self.landing_defaults)
   except (ValueError,TypeError) as exc:raise RequestError(str(exc)) from exc
   use_cache=body.get('prefix_cache',self.prefix_cache)
   if type(use_cache) is not bool:raise RequestError('prefix_cache must be a boolean')
   body=dict(body);body.pop('thinking_budget',None);body.pop('thinking_loop',None);body.pop('prefix_cache',None)
   if not chat and (body.get('response_format') or {}).get('type')=='json_object':raise RequestError('json_object is qualified only for chat completions')
   if chat and has_images(body.get("messages",[])):
    if frontend is None:raise RequestError("Vision frontend unavailable")
    prepared=frontend.prepare(body,self)
   else:prepared=super()._prepare(body,chat)
   # Tabby adds BOS by default. Keep v43 raw-ID pairing rather than upstream HTTP's no-BOS encoding.
   bos=1  # Pinned Tabby backend BOS, verified against all 26 frozen v43 cases
   if prepared.prompt[0]!=bos:prepared.prompt.insert(0,bos)
   if len(prepared.prompt)>MAX_PROMPT:raise RequestError(f'Prompt exceeds the {MAX_PROMPT:,}-token input ceiling')
   if prepared.max_tokens>MAX_OUTPUT:raise RequestError('max_tokens exceeds the 16,384-token serving cap')
   prepared.max_tokens=min(prepared.max_tokens,CONTEXT-len(prepared.prompt))
   if prepared.grammar is not None and (prepared.think_budget or initial_thinking(tok,prepared.prompt)!=prepared.thinking):
    raise RequestError('JSON requires an unambiguous fresh assistant reasoning boundary and no legacy gate')
   prepared.landing_config=config;prepared.prefix_cache=use_cache
   if config.enabled:
    try:prepared.landing_state=ThinkingLanding(config,tok.token_to_id('<think>'),tok.token_to_id('</think>'),tok.encode(LANDING_TEXT,add_special_tokens=False).ids,prepared.max_tokens,initial_thinking=initial_thinking(tok,prepared.prompt),logger=lambda event:print(event,flush=True),answer_prefix_ids=())
    except (ValueError,TypeError) as exc:raise RequestError(str(exc)) from exc
   return prepared
 import tensorfold.cuda.server as upstream
 namespace=dict(upstream.__dict__);exec(compile((HERE/'http_run.py').read_text(),str(HERE/'http_run.py'),'exec'),namespace)
 ServingApp.run=namespace['run']
 app=ServingApp(Adapter(),model,'glm53',default_thinking=True,sampling={'temperature':0.3,'top_p':0.95,'top_k':0,'min_p':0.05},max_tokens=MAX_OUTPUT,context_window=CONTEXT,reasoning_effort='high',thinking_budget=0)
 from tensorfold.engine.thinking_landing import LandingConfig
 app.landing_defaults=LandingConfig(**json.loads((HERE/'thinking_defaults.json').read_text()))
 serving=json.loads((HERE/'serving_defaults.json').read_text())
 if set(serving)!={'prefix_cache','keepalive_seconds'} or type(serving['prefix_cache']) is not bool or type(serving['keepalive_seconds']) not in (int,float) or serving['keepalive_seconds']<0:
  raise ValueError('serving_defaults.json needs a boolean prefix_cache and a non-negative keepalive_seconds')
 app.prefix_cache=serving['prefix_cache'];app.keepalive_seconds=serving['keepalive_seconds']
 return app

def reserve_guard(m,a,reserve):
 """The inherited v42 guard with this copy's host reserve; every other limit is unchanged."""
 GIB=m.GIB
 class Guard(m.Guard):
  def __call__(self,phase,forecast=0):
   try:
    paths=[l.split(':',2)[2] for l in (self.proc/'self/cgroup').read_text().splitlines() if l.startswith('0::')]
    if len(paths)!=1 or '..' in Path(paths[0]).parts:raise ValueError('unresolved cgroup')
    own=self.cg/paths[0].lstrip('/')
    vmax=(own/'memory.max').read_text().strip();smax=(own/'memory.swap.max').read_text().strip()
    used=int((own/'memory.current').read_text());swap=int((own/'memory.swap.current').read_text())
    fields={l.split(':')[0]:int(l.split()[1])*1024 for l in (self.proc/'meminfo').read_text().splitlines() if l.startswith(('MemAvailable:','SwapTotal:','SwapFree:'))}
    hs=fields['SwapTotal']-fields['SwapFree'];growth=hs-self.a.host_swap_baseline_bytes
    r=dict(phase=phase,host_available=fields['MemAvailable'],host_swap=hs,host_swap_growth=growth,cgroup_current=used,cgroup_max=vmax,cgroup_swap_max=smax,cgroup_swap_current=swap,forecast_bytes=forecast)
    self.events.append(r)
    if smax!='0' or swap!=0 or vmax=='max' or not 0<int(vmax)<=95*GIB:raise ValueError('finite95GiB/zero-cgroup-swap contract')
    if fields['MemAvailable']<reserve or growth>GIB:raise ValueError('%dGiB reserve or baseline+1GiB swap breach'%(reserve//GIB))
    if forecast and (forecast+3*GIB>int(vmax) or forecast+reserve+3*GIB>fields['MemAvailable']):raise m.Unsupported('source_forecast_cannot_preserve_reserve_before_load')
    return r
   except m.Unsupported:raise
   except Exception as ex:raise m.SafetyAbort(str(ex)) from ex
 return Guard(a)

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--host-swap-baseline-bytes',type=int,required=True);a=p.parse_args()
 m=import_path('serve_v42_readonly',ROOT/'context-v42/tensorfold/runner.py');m.STAGE=STAGE;m.ORIGINAL=COPY/'src'
 a.stage=STAGE;a.authorized_guarded_gpu=True;m.authorize(a)
 guard=reserve_guard(m,a,RESERVE_GIB*m.GIB);guard.events=collections.deque(maxlen=256);guard('serving_startup')
 ns=dict(m.__dict__,MEASURED85=True,MEASURED_PEAK=PEAK_ESTIMATE)
 exec(compile((ROOT/'accuracy-v43/runtime_loader.py').read_text(),'v43_owned_loader','exec'),ns)
 runtime=ns['load_runtime'](a,{},dict(cache_slots=SLOTS,dense_limit=2051,long_context=True),guard)
 e,tok,torch,api,Sampling,scratch,forward,hooks,admission=runtime
 if not Path(api.__file__).resolve().is_relative_to((COPY/'src').resolve()):raise RuntimeError('Original TensorFold imported instead of source copy')
 with torch.inference_mode():
  warm=dict(id='warmup',raw_prompt_ids=[tok.token_to_id('<|endoftext|>'),*tok.encode('warmup',add_special_tokens=False).ids],max_new_tokens=4,force_length=True,sampler={'temperature':0},seed=0)
  m.collect(e,tok,warm,torch,api,Sampling,scratch,guard)
  from tensorfold.families.glm5_next.cuda.graphs import Graphs
  for b in (e.buf,e.pbuf,e.mbuf):
   if b is not None:
    for n in (1,2):b.parents(n)
    b.ids.zero_()
    if b.hin is not None:b.hin.zero_()
  e.reset();torch.cuda.synchronize();e.graphs=Graphs(e,main_rows=(1,2),mtp_rows=(1,2));e.reset();torch.cuda.synchronize();guard('after_serving_graph_capture')
 from vision import Frontend
 guard("before_vision_loader");frontend=Frontend(a.model,runtime);guard("after_vision_loader")
 app=make_app(runtime,a.model,frontend)
 print(json.dumps(dict(status='ready',model='glm53',admission=admission,admission_note='peak is the 89,344-slot measurement plus a 3.5 GiB estimate for 262,144 slots',reserve_gib=RESERVE_GIB,context=CONTEXT,max_prompt=MAX_PROMPT,max_output=MAX_OUTPUT,text_only=False,vision=True,tools=True,default_sampling=app.sampling,reasoning_effort=app.reasoning_effort,thinking_loop=dataclasses.asdict(app.landing_defaults),prefix_cache=app.prefix_cache,keepalive_seconds=app.keepalive_seconds,tensorfold_source=str(COPY))),flush=True)
 from tensorfold.cuda.http import serve
 serve(app,'127.0.0.1',8002)
if __name__=='__main__':main()
