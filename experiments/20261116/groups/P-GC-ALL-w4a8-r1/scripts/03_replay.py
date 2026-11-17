import json,pathlib,subprocess,time,urllib.request,sys
r=pathlib.Path('/root/dsk_opt_goal_20261116/groups/P-GC-ALL-w4a8-r1')
model='sf-finance-dsk41-dynamo-vllm-alyeast-prod'
pid=json.loads((r/'evidence/service-pids.json').read_text())['worker']
assert pathlib.Path('/root/dsk_opt_goal_20261116/evidence/gc-service-adapter-exit.txt').read_text().strip()=='0'
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

gc_files=list((r/'evidence/gc').glob('rank*.json'))
assert len(gc_files)==8
for file in gc_files:
    d=json.loads(file.read_text())
    assert d['target_count']==40 and len(d['target_layers'])==40
    assert d['finalize']==d['compact_indexer']==d['headrow']=='1'
    assert all(x['backend']=='gc_moe.sm90_fp8_fp4_gc_moe' and x['output_reduced'] for x in d['target_layers'])
for file in audit_files:
    layers=json.loads(file.read_text())
    assert sum(x['scope']=='target' for x in layers)==0
    assert sum(x['scope']=='draft' for x in layers)==3
(r/'evidence/activation-dtype-gate.json').write_text(json.dumps({'passed':True,'target':'GC native FP8 E4M3 K128, MXFP4 weights; 40 layers/rank','draft':'native unchanged','humming_target_layers':0,'finalize':'GC output BF16 scale + shared add fusion'},indent=2))
assert len(list((r/'evidence/headrow').glob('headrow-rank*.json')))==8
assert len(list((r/'evidence/compact').glob('workspace-rank*.json')))==8
# One isolated synthetic preflight: distinct user-only prompt, no dataset prefix.
payload={'model':model,'messages':[{'role':'user','content':'Adapter preflight 20261116. Summarize these synthetic numbers in several sentences: '+ ' '.join(str(i) for i in range(1024))}],'max_tokens':32,'temperature':0,'stream':False}
request=urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
with urllib.request.urlopen(request,timeout=180) as response: preflight_result=json.load(response)
(r/'evidence/preflight.json').write_text(json.dumps({'purpose':'kernel-branch readiness; excluded from150 timing; synthetic non-dataset prompt','usage':preflight_result.get('usage'),'finish_reasons':[x.get('finish_reason') for x in preflight_result.get('choices',[])]},indent=2))
assert len(list((r/'evidence/compact').glob('compact-branch-rank*.json')))==8, 'Compact path not selected in startup; refusing timed replay'

(r/'evidence/audit-models-ready.json').write_text(json.dumps(models,indent=2))
cmd=[sys.executable,str(pathlib.Path('/root/dsk_opt_goal_20261116/scripts/09_replay150.py')),'--save-responses','--plan','/root/bench_dsk_first150_20261116/first150.plan.json.gz','--base-url','http://127.0.0.1:8000/v1','--model',model,'--output',str(r/'results/replay150.jsonl')]
with (r/'logs/replay150.log').open('x') as log: code=subprocess.call(cmd,stdout=log,stderr=subprocess.STDOUT)
(r/'evidence/replay-exit.json').write_text(json.dumps({'exit_code':code,'command':cmd}))
sys.exit(code)
