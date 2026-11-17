import pathlib,subprocess,json,os,signal,time,socket,re,urllib.request,traceback
R=pathlib.Path('/root/dsk_opt_goal_20261116');G=R/'groups/E1-native-r1';OLD=R/'groups/P-ALL-w4a8-stride-r2';E=pathlib.Path('/root/gc_moe_isolated_20261116');owned={}
assert json.loads((E/'evidence/runtime-import-smoke.json').read_text())['passed']
assert (E/'evidence/runtime-import-smoke-main-aligned-exit.txt').read_text().strip()=='0'
assert not (G/'evidence/run-state.json').exists(),'Never overwrite a run'
def state(status,**kwargs):
 (G/'evidence/run-state.json').write_text(json.dumps({'status':status,'pids':{k:v.pid for k,v in owned.items()},'time_unix':time.time(),**kwargs},indent=2))
def launch(name,script):
 with (G/f'logs/{name}.log').open('x') as f:owned[name]=subprocess.Popen(['bash',str(G/'scripts'/script)],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 (G/'evidence/service-pids.json').write_text(json.dumps({k:v.pid for k,v in owned.items()},indent=2))
try:
 assert json.loads((OLD/'evidence/replay-exit.json').read_text())['exit_code']==0
 ns=re.search(r'^export DYN_NAMESPACE=(.+)$',(OLD/'scripts/00_runtime_env.sh').read_text(),re.M).group(1)
 for role in ['worker','frontend']:
  pid=json.loads((OLD/'evidence/service-pids.json').read_text())[role];p=pathlib.Path(f'/proc/{pid}')
  if not p.exists():continue
  env=dict(x.split('=',1) for x in (p/'environ').read_bytes().decode().split('\0') if '=' in x)
  assert env.get('DYN_NAMESPACE')==ns and os.getpgid(pid)==pid,'Previous PID ownership mismatch'
  os.killpg(pid,signal.SIGTERM)
 state('waiting_for_gpu_release')
 for _ in range(36):
  out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=10)
  if not out.strip():break
  time.sleep(5)
 else:raise RuntimeError('GPU processes remain; do not kill unknown processes')
 for port in [8000,8081]:
  with socket.socket() as sock:assert sock.connect_ex(('127.0.0.1',port))!=0,('Port busy',port)
 launch('frontend','02_frontend.sh');launch('worker','01_worker.sh');state('loading_native_E1')
 deadline=time.monotonic()+7200
 while time.monotonic()<deadline:
  for role,p in owned.items():
   if p.poll() is not None:raise RuntimeError(f'{role} exited {p.returncode}')
  try:
   with urllib.request.urlopen('http://127.0.0.1:8000/v1/models',timeout=5) as response:models=json.load(response)
   if any(x.get('id')=='sf-finance-dsk41-dynamo-vllm-alyeast-prod' for x in models.get('data',[])):break
  except Exception:pass
  time.sleep(5)
 else:raise RuntimeError('Startup exceeded 2h deadline')
 (G/'evidence/models.json').write_text(json.dumps(models,indent=2));state('short_request')
 payload={'model':'sf-finance-dsk41-dynamo-vllm-alyeast-prod','messages':[{'role':'user','content':'请只回复 OK。'}],'max_tokens':32,'temperature':0,'stream':False}
 req=urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=300) as response:answer=json.load(response)
 (G/'evidence/preflight.json').write_text(json.dumps({'scope':'service compatibility smoke only, excluded from formal150','request':payload,'response':answer},ensure_ascii=False,indent=2))
 assert answer.get('choices'),answer
 files=list((G/'evidence/loaded-layer-configs').glob('rank*.json'));assert len(files)==8
 for file in files:
  a=json.loads(file.read_text());assert sum(x['scope']=='target' for x in a)==40 and sum(x['scope']=='draft' for x in a)==3
 state('short_request_passed_service_retained');print('E1 native full model and short request passed',flush=True)
except BaseException as exc:
 state('failed',error=repr(exc));traceback.print_exc()
 for p in owned.values():
  if p.poll() is None:
   try:os.killpg(p.pid,signal.SIGTERM)
   except ProcessLookupError:pass
 raise
