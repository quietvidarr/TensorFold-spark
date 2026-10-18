"""Live depth check: needle recall and decode rate from a short prompt to near the context ceiling.

Run on the host against the internal port once this copy is serving:
    python3 -B serving/depth_check.py [--url http://127.0.0.1:8002] [--out runs/depth-check.json] [--depths short,8k,...]

Per depth, one conversation:
  1. a document of numbered records holding five planted access codes, and a question asking for all five
     (greedy, thinking on). The prompt is filled from a reset, so this is the fresh time to first token.
  2. three identical follow-up turns asking for a long description (greedy, 512 tokens, reasoning counted).
     They resume from turn 1's prompt state, so their decode rate is measured at depth without refilling it.
Results are written after every depth. Standard library only.
"""
import argparse,json,re,sys,time,urllib.request
from pathlib import Path

DEPTHS=dict(short=0,**{'8k':385,'32k':1540,'85k':4050,'128k':6100,'200k':9600,'250k':12000})
ORDER=['short','8k','32k','85k','128k','200k','250k']

def record(i):
 return f'Record {i:05d}: station {i%97:02d} logged value {(i*7919)%10007} at tick {i*13}.'

def document(lines,salt):
 codes={k:f'{(salt*7717+k*104729)%900000+100000}' for k in range(1,6)}
 at={max(1,int(lines*f)):k for k,f in zip(range(1,6),(0.1,0.3,0.5,0.7,0.9))} if lines else {}
 out=[]
 for i in range(lines):
  out.append(record(i))
  if i+1 in at:out.append(f'Note: the access code for vault {at[i+1]} is {codes[at[i+1]]}.')
 if not lines:out=[f'Note: the access code for vault {k} is {codes[k]}.' for k in range(1,6)]
 return '\n'.join(out),codes

def ask(url,body):
 began=time.monotonic()
 reply=json.load(urllib.request.urlopen(urllib.request.Request(url+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=7200))
 stats=reply['tensorfold'];usage=reply['usage'];message=reply['choices'][0]['message']
 return dict(content=message.get('content') or '',finish=reply['choices'][0]['finish_reason'],prompt_tokens=usage['prompt_tokens'],
             completion_tokens=usage['completion_tokens'],reasoning_tokens=usage['completion_tokens_details']['reasoning_tokens'],
             cached=stats.get('cached'),prefill_s=round(stats['prefill_s'],3),decode_s=round(stats['decode_s'],3),
             decode_tok_s=None if stats.get('decode_tok_s') is None else round(stats['decode_tok_s'],3),
             drafted=stats.get('drafted'),accepted=stats.get('accepted'),sparse_graph_replays=stats.get('sparse_graph_replays'),
             peak_reserved=stats.get('peak_reserved'),trigger=(stats.get('thinking_loop') or {}).get('trigger'),wall_s=round(time.monotonic()-began,3))

def available():
 return next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))

def main():
 p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8002');p.add_argument('--out',default='');p.add_argument('--depths',default=','.join(ORDER));a=p.parse_args()
 rows=[]
 for index,name in enumerate(a.depths.split(',')):
  text,codes=document(DEPTHS[name],index+1)
  first=[dict(role='user',content=text+'\n\nList the access codes for vaults 1 to 5 as one JSON object that maps each vault number to its code. Reply with the JSON object only.')]
  needle=ask(a.url,dict(model='glm53',messages=first,max_tokens=4096,temperature=0,stream=False,prefix_cache=False))
  found=dict(re.findall(r'"?(\d)"?\s*:\s*"?(\d{6})"?',needle['content']))
  correct=sum(found.get(str(k))==v for k,v in codes.items())
  follow=first+[dict(role='assistant',content=needle['content']),
                dict(role='user',content='Describe in plain prose, at length, how you would audit records like these for errors. Keep writing until you are stopped.')]
  # the landing guard refuses caps under its 1,024-token answer reserve, so it is off for these 512-token turns
  # thinking stays on: switching it off changes how the template renders the history, and the prompt would no longer extend turn 1
  body=dict(model='glm53',messages=follow,max_tokens=512,temperature=0,stream=False,thinking_loop=dict(enabled=False))
  runs=[ask(a.url,body) for _ in range(3)]
  tokens=sum(r['completion_tokens']-1 for r in runs);seconds=sum(r['decode_s'] for r in runs)
  row=dict(depth=name,prompt_tokens=needle['prompt_tokens'],fresh_prefill_s=needle['prefill_s'],fresh_cached=needle['cached'],
           fresh_prefill_tok_s=round(needle['prompt_tokens']/needle['prefill_s'],1) if needle['prefill_s'] else None,
           needles_correct=correct,needles_total=5,needle_finish=needle['finish'],needle_reasoning_tokens=needle['reasoning_tokens'],
           needle_trigger=needle['trigger'],needle_decode_tok_s=needle['decode_tok_s'],needle_answer=needle['content'][:400],
           decode_prompt_tokens=runs[0]['prompt_tokens'],decode_cached=[r['cached'] for r in runs],decode_prefill_s=[r['prefill_s'] for r in runs],
           decode_completion_tokens=[r['completion_tokens'] for r in runs],decode_reasoning_tokens=[r['reasoning_tokens'] for r in runs],
           decode_finish=[r['finish'] for r in runs],decode_tok_s=[r['decode_tok_s'] for r in runs],
           decode_tok_s_weighted=round(tokens/seconds,3) if seconds else None,
           draft_acceptance=round(sum(r['accepted'] or 0 for r in runs)/max(1,sum(r['drafted'] or 0 for r in runs)),4),
           sparse_graph_replays=runs[-1]['sparse_graph_replays'],peak_reserved=max(r['peak_reserved'] or 0 for r in runs+[needle]),
           host_available_after=available(),time_ns=time.time_ns())
  rows.append(row);print(json.dumps(row),flush=True)
  if a.out:Path(a.out).write_text(json.dumps(dict(status='running',url=a.url,contract='greedy; needle turn thinking on, 4096 cap, fresh prefill; decode turns thinking on, landing guard off, 512 tokens (reasoning included), 3 identical runs resumed from the needle prompt state',depths=rows),indent=2)+'\n')
 if a.out:
  data=json.loads(Path(a.out).read_text());data['status']='complete';Path(a.out).write_text(json.dumps(data,indent=2)+'\n')
 return 0
if __name__=='__main__':sys.exit(main())
