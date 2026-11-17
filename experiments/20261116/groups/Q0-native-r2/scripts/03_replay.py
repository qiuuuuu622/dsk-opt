import json,pathlib,subprocess,time,urllib.request,sys
r=pathlib.Path('/root/dsk_opt_goal_20261116/groups/Q0-native-r2')
model='sf-finance-dsk41-dynamo-vllm-alyeast-prod'
pid=json.loads((r/'evidence/service-pids.json').read_text())['worker']
start=time.monotonic()
while time.monotonic()-start<2400:
 if not pathlib.Path(f'/proc/{pid}').exists(): raise RuntimeError('Audit worker exited before ready')
 try:
  with urllib.request.urlopen('http://127.0.0.1:8000/v1/models',timeout=3) as response: models=json.load(response)
  ready=any(x['id']==model for x in models.get('data',[]))
 except Exception: ready=False
 if ready: break
 time.sleep(10)
else: raise TimeoutError('Model not ready within 40 minutes')
audit_files=list((r/'evidence/loaded-layer-configs').glob('rank*.json'))
if len(audit_files)!=8: raise RuntimeError('Expected 8 post-load metadata files')
if any(not json.loads(p.read_text()) for p in audit_files): raise RuntimeError('Empty layer audit')
(r/'evidence/audit-models-ready.json').write_text(json.dumps(models,indent=2))
cmd=[sys.executable,str(pathlib.Path('/root/dsk_opt_goal_20261116/scripts/28_quality_replay150.py')),'--temperature','0','--seed','1234','--save-responses','--plan','/root/bench_dsk_first150_20261116/first150.plan.json.gz','--base-url','http://127.0.0.1:8000/v1','--model',model,'--output',str(r/'results/replay150.jsonl')]
with (r/'logs/replay150.log').open('x') as log: code=subprocess.call(cmd,stdout=log,stderr=subprocess.STDOUT)
(r/'evidence/replay-exit.json').write_text(json.dumps({'exit_code':code,'command':cmd}))
sys.exit(code)
