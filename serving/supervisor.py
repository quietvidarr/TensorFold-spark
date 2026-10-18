"""Persistent serving supervisor. Original displaced units unchanged; rollback on stop/failure.
Live15GiB reserve, zero container swap, immutable host baseline+1GiB growth.
"""
import argparse,fcntl,hashlib,importlib.util,json,os,signal,subprocess,sys,time,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent;STUDY=HERE.parent.parent;ROOT=STUDY.parent;NAME='tfglm-thinking-v59-pipeline';GIB=1024**3;RESERVE=15*GIB  # host memory this copy keeps free
IMAGE='sha256:1f626b72bd7b20a470dae03ae18039049d50a0d496de20bf818b3b612c8699ee'
def dump(p,x):
 t=p.with_suffix('.pending');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--execute-authorized-serving',action='store_true');p.add_argument('--host-swap-baseline-bytes',type=int);a=p.parse_args()
 if not a.execute_authorized_serving:raise ValueError('Explicit serving authorization required')
 run=STUDY/'runs/live-serving';run.mkdir(parents=True,exist_ok=True);lock=(run/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 spec=importlib.util.spec_from_file_location('maintenance_readonly',ROOT/'scripts/maintenance_controller.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 ctl=m.Controller(run);child=None;cg=None;baseline=None;minimum=None;maxgrowth=0;ready=False
 def event(phase,**data):ctl.event(phase,**data)
 def fail_signal(signum,frame):raise RuntimeError('Serving supervisor stopped: '+str(signum))
 for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,fail_signal)
 try:
  ctl.snapshot={u:{'properties':m.properties(u),'unit_sha256':hashlib.sha256(m.cmd(['systemctl','--user','cat',u]).encode()).hexdigest()}for u in m.UNITS}
  if ctl.snapshot[m.UNITS[0]]['properties']['ActiveState']!='active':raise RuntimeError('Qwen not active before cutover')
  if m.inspect_container(NAME,timeout=3)['exists']:raise RuntimeError('Owned serving container already exists')
  m.idle();dump(run/'audit-maintenance-snapshot.json',ctl.snapshot);event('idle_verified')
  baseline=m.host_sample();minimum=baseline['mem_available']
  if a.host_swap_baseline_bytes is not None:
   if a.host_swap_baseline_bytes<0:raise ValueError('Negative inherited swap baseline')
   baseline['host_swap_used']=a.host_swap_baseline_bytes
  event('prestop_baseline',sample=baseline)
  ctl.changed=True
  for u in m.UNITS[1:]:m.cmd(['systemctl','--user','stop',u])
  m.idle();m.cmd(['systemctl','--user','stop',m.UNITS[0]])
  if int(m.properties(m.UNITS[0])['MainPID']):raise RuntimeError('Displaced GPU PID remains')
  event('load_baseline',sample=m.host_sample(),immutable_host_swap_baseline=baseline['host_swap_used'])
  if baseline['mem_available']<RESERVE:raise RuntimeError('Reserve before launch')
  argv=['docker','run','--pull','never','--name',NAME,'--label','tfglm.stage='+NAME,'--gpus','all','--network','host','--memory',str(95*GIB),'--memory-swap',str(95*GIB),'--entrypoint','python3','-v',str(ROOT)+':/work:ro','-v',str(STUDY)+':/work/thinking-v59:rw','-v',str(Path.home()/'models/glm53-flash-exl3-2.05bpw')+':/model:ro']
  for v in ['EXL3_INT8_GEMV=2','PYTHONDONTWRITEBYTECODE=1','TFGLM_CONTROLLER_STAGE='+NAME,'TORCH_EXTENSIONS_DIR=/work/thinking-v59/runs/torch_extensions','PYTHONPATH=/work/thinking-v59/grammar-deps:/work/thinking-v59/TensorFold/src:/opt/sglang/lib/python3.12/site-packages']:argv+=['-e',v]
  argv += [IMAGE,'-B','-S','/work/thinking-v59/TensorFold/serving/server.py','--model','/model','--host-swap-baseline-bytes',str(baseline['host_swap_used'])]
  ctl.container=NAME;so=(run/'stdout.log').open('w');se=(run/'stderr.log').open('w');child=subprocess.Popen(argv,stdout=so,stderr=se);start=time.monotonic();samples=0;lastwrite=0
  while True:
   if child.poll() is not None:raise RuntimeError('Serving process exited '+str(child.returncode))
   owner=m.inspect_container(NAME,timeout=2)
   if owner['running'] and owner['pid']:
    cg=next(l.split('::',1)[1]for l in Path('/proc/'+str(owner['pid'])+'/cgroup').read_text().splitlines()if l.startswith('0::'))
    sample=m.host_sample(cg)
    if any(k not in sample for k in ('memory.swap.current','memory.swap.max')):raise RuntimeError('Swap evidence missing')
    if sample['memory.swap.max']!='0' or int(sample['memory.swap.current']):raise RuntimeError('Container swap guard')
   elif time.monotonic()-start>10:raise RuntimeError('Serving container absent after10seconds')
   else:sample=m.host_sample()
   samples+=1;minimum=min(minimum,sample['mem_available']);maxgrowth=max(maxgrowth,sample['host_swap_used']-baseline['host_swap_used'])
   if sample['mem_available']<RESERVE or maxgrowth>GIB:raise RuntimeError('Serving live reserve or host swap guard')
   if time.monotonic()-lastwrite>=2:
    state=dict(status='serving' if ready else 'loading',model='glm53',backend='tensorfold',sample=sample,samples=samples,min_available=minimum,max_host_swap_growth=maxgrowth,host_swap_baseline=baseline['host_swap_used'],reserve_bytes=RESERVE,cgroup_swap_max=0,host_swap_growth_limit=GIB)
    dump(run/'state.json',state);lastwrite=time.monotonic()
   if not ready:
    try:
     models=json.load(urllib.request.urlopen('http://127.0.0.1:8002/v1/models',timeout=.5))
     if [d['id']for d in models['data']]!=['glm53']:raise RuntimeError('Wrong served identity')
     ready=True
     # Isolated quality run: parent owns ingress; do not admit public traffic.
     event('serving_ready',model=models,container_pid=owner['pid'],sample=sample)
    except (OSError,ValueError):pass
    if not ready and time.monotonic()-start>1200:raise RuntimeError('Readiness timeout')
   time.sleep(.5)
 except BaseException as ex:
  try:event('serving_stopped',error=type(ex).__name__+': '+str(ex))
  except OSError:pass
  raise
 finally:
  for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,signal.SIG_IGN)
  ctl.restore()
  if child is not None:
   child.wait(timeout=15)
   m.cmd(['docker','rm',NAME],check=True,timeout=10)
  if baseline is not None:dump(run/'state.json',dict(status='stopped_qwen_restored',minimum_available=minimum,max_host_swap_growth=maxgrowth))
if __name__=='__main__':main()
