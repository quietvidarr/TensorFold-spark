"""Live check that a resumed prompt gives a fresh prompt's reply. Standard library only.

Run on the host against the internal port once this copy is serving:
    python3 -B serving/cache_equivalence.py [--url http://127.0.0.1:8002] [--out runs/cache-equivalence.json]

Each case sends a first turn, then the next turn three ways with one seed: resumed from the first turn's
state, resent (resumed again from that same earlier state), and with ``prefix_cache: false``. The three
replies must hold the same token ids. Cases sit below, across and above the 2,051-token dense limit.
"""
import argparse,json,sys,time,urllib.request
from pathlib import Path

def filler(lines):
 return '\n'.join(f'Record {i:05d}: station {i%97:02d} logged value {(i*7919)%10007} at tick {i*13}.' for i in range(lines))

CASES=[dict(name='short',lines=0,grow=0),
       dict(name='across_dense_limit',lines=60,grow=70),
       dict(name='long_context',lines=520,grow=20)]

def ask(url,messages,seed,max_tokens,cache):
 body=dict(model='glm53',messages=messages,max_tokens=max_tokens,seed=seed,temperature=0.3,stream=False,return_token_ids=True,prefix_cache=cache)
 began=time.monotonic()
 reply=json.load(urllib.request.urlopen(urllib.request.Request(url+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=1800))
 stats=reply['tensorfold'];usage=reply['usage']
 cached=(usage.get('prompt_tokens_details') or {}).get('cached_tokens',stats.get('cached'))
 return dict(ids=stats['token_ids'],content=reply['choices'][0]['message'].get('content') or '',finish=reply['choices'][0]['finish_reason'],
             prompt_tokens=usage['prompt_tokens'],cached=cached,engine_cached=stats.get('cached'),prefill_s=round(stats['prefill_s'],3),
             wall_s=round(time.monotonic()-began,3),trigger=(stats.get('thinking_loop') or {}).get('trigger'))

def first_difference(a,b):
 return next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),None if len(a)==len(b) else min(len(a),len(b)))

def main():
 p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8002');p.add_argument('--out',default='');p.add_argument('--max-tokens',type=int,default=2400);a=p.parse_args()
 results=[];passed=True
 for index,case in enumerate(CASES):
  seed=7100+index;notes=filler(case['lines'])
  first=[dict(role='system',content='You are a careful assistant. Keep replies to one short sentence.'),
         dict(role='user',content=(notes+'\n\n' if notes else '')+'Reply with the single word: ready')]
  one=ask(a.url,first,seed,a.max_tokens,True)
  more=filler(case['grow'])
  second=first+[dict(role='assistant',content=one['content']),
                dict(role='user',content=(more+'\n\n' if more else '')+'Now reply with the single word: done')]
  resumed=ask(a.url,second,seed+50,a.max_tokens,True)
  resent=ask(a.url,second,seed+50,a.max_tokens,True)
  fresh=ask(a.url,second,seed+50,a.max_tokens,False)
  row=dict(case=case['name'],first_prompt_tokens=one['prompt_tokens'],second_prompt_tokens=resumed['prompt_tokens'],
           resumed_cached=resumed['engine_cached'],resent_cached=resent['engine_cached'],fresh_cached=fresh['engine_cached'],
           resumed_prefill_s=resumed['prefill_s'],resent_prefill_s=resent['prefill_s'],fresh_prefill_s=fresh['prefill_s'],
           reply_tokens=len(fresh['ids']),finish=[resumed['finish'],resent['finish'],fresh['finish']],
           triggers=[resumed['trigger'],resent['trigger'],fresh['trigger']],
           resumed_equals_fresh=resumed['ids']==fresh['ids'],resent_equals_fresh=resent['ids']==fresh['ids'],
           first_difference=[first_difference(resumed['ids'],fresh['ids']),first_difference(resent['ids'],fresh['ids'])],
           resumed_from_first_turn=resumed['engine_cached']==one['prompt_tokens'],resent_from_first_turn=resent['engine_cached']==one['prompt_tokens'],
           fresh_filled_everything=fresh['engine_cached']==0)
  ok=all(row[k] for k in ('resumed_equals_fresh','resent_equals_fresh','resumed_from_first_turn','resent_from_first_turn','fresh_filled_everything'))
  row['passed']=ok;passed=passed and ok;results.append(row);print(json.dumps(row),flush=True)
 summary=dict(status='passed' if passed else 'failed',url=a.url,cases=results,time_ns=time.time_ns())
 if a.out:Path(a.out).write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps(dict(status=summary['status'])),flush=True)
 return 0 if passed else 1
if __name__=='__main__':sys.exit(main())
