"""Run one pre-created formal group; retain service for post-run diagnostics."""
import argparse,json,os,pathlib,re,signal,socket,subprocess,time,traceback
ROOT=pathlib.Path('/root/dsk_opt_goal_20261116')
PYTHON=str(ROOT/'.venv-e0/bin/python')
p=argparse.ArgumentParser();p.add_argument('--group',required=True);p.add_argument('--stop-group',required=True);a=p.parse_args()
assert a.group in ('P-GC-ALL-w4a8-r1','P-FAST-GC-ALL-r1')
assert re.fullmatch(r'[A-Za-z0-9-]+',a.stop_group)
g=ROOT/'groups'/a.group;old=ROOT/'groups'/a.stop_group
state_path=g/'evidence/run-state.json'
assert not state_path.exists(),'Never overwrite a run'
owned={}
def state(status,**kwargs):
 state_path.write_text(json.dumps({'status':status,'group':a.group,'time_unix':time.time(),'pids':{k:v.pid for k,v in owned.items()},**kwargs},indent=2))
def stop_old():
 exit_status=json.loads((old/'evidence/replay-exit.json').read_text())
 assert exit_status['exit_code']==0,'Previous replay must finish before switching'
 namespace=re.search(r'^export DYN_NAMESPACE=p-fast-gc-all-r1
 for key in ['worker','frontend']:
  pid=json.loads((old/'evidence/service-pids.json').read_text())[key]
  proc=pathlib.Path(f'/proc/{pid}')
  if not proc.exists():continue
  env=dict(x.split('=',1) for x in (proc/'environ').read_bytes().decode().split('\0') if '=' in x)
  assert env.get('DYN_NAMESPACE')==namespace,(key,'PID namespace mismatch')
  assert os.getpgid(pid)==pid
  os.killpg(pid,signal.SIGTERM)
 for _ in range(36):
  try:
   out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=10)
   if not out.strip():return
  except subprocess.TimeoutExpired:pass
  time.sleep(5)
 raise RuntimeError('GPUs not idle after previous service stopped')
def launch(key,command,logname):
 with (g/'logs'/logname).open('x') as log:
  owned[key]=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 return owned[key]
try:
 assert (ROOT/'evidence/gc-service-adapter-triton37-exit.txt').read_text().strip()=='0'
 assert (ROOT/'evidence/indexer-strided-e1-exit.txt').read_text().strip()=='0'
 from e1_timing_guard import assert_timing_clear
 assert_timing_clear()
 state('stopping_previous')
 stop_old()
 for port in [8000,8081]:
  with socket.socket() as s:assert s.connect_ex(('127.0.0.1',port))!=0,('Port busy',port)
 launch('frontend',['bash',str(g/'scripts/02_frontend.sh')],'frontend.log')
 launch('worker',['bash',str(g/'scripts/01_worker.sh')],'worker.log')
 (g/'evidence/service-pids.json').write_text(json.dumps({k:v.pid for k,v in owned.items()},indent=2))
 replay=launch('replay_launcher',[PYTHON,str(g/'scripts/03_replay.py')],'launcher.log')
 (g/'evidence/service-pids.json').write_text(json.dumps({k:v.pid for k,v in owned.items()},indent=2))
 state('startup_and_replay')
 deadline=time.monotonic()+6500
 while replay.poll() is None:
  for key in ['worker','frontend']:
   if owned[key].poll() is not None:raise RuntimeError(f'{key} exited: {owned[key].returncode}')
  if time.monotonic()>deadline:raise TimeoutError('Startup and replay exceeded limit')
  time.sleep(5)
 assert replay.returncode==0,('Replay failed',replay.returncode)
 with (g/'logs/validation.log').open('x') as log:subprocess.run([PYTHON,str(g/'scripts/04_validate.py')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
 summary=json.loads((g/'results/replay150.summary.json').read_text());assert summary['success']==150 and summary['failed']==0
 state('completed_service_retained',summary=summary)
 print(json.dumps({'group':a.group,'status':'completed','summary':summary}),flush=True)
except BaseException as exc:
 state('failed',error=f'{type(exc).__name__}: {exc}')
 for proc in owned.values():
  if proc.poll() is None:
   try:os.killpg(proc.pid,signal.SIGTERM)
   except ProcessLookupError:pass
 traceback.print_exc();raise
