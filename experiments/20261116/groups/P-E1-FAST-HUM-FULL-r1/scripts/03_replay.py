import json,pathlib,subprocess,time,urllib.request,sys
r=pathlib.Path('/root/dsk_opt_goal_20261116/groups/P-E1-FAST-HUM-FULL-r1')
model='sf-finance-dsk41-dynamo-vllm-alyeast-prod'
pid=json.loads((r/'evidence/service-pids.json').read_text())['worker']
gate=json.loads(pathlib.Path('/root/dsk_opt_goal_20261116/groups/QH1-a8-r1/evidence/quality-gate.json').read_text())
assert gate['status']=='structural_proxy_pass',gate
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

counts = {'target': 0, 'draft': 0}
for file in audit_files:
    layers = json.loads(file.read_text())
    assert sum(x['scope']=='target' for x in layers)==40
    assert sum(x['scope']=='draft' for x in layers)==3
    for layer in layers:
        scope=layer['scope']
        for config in layer['humming_configs'].values():
            a=config['a_dtype']; b=config['b_dtype']
            expected=(8,4,3) if scope=='target' else (16,8,7)
            assert (a['num_bits'],a['exponent_bits'],a['mantissa_bits'])==expected,(file,layer['name'],a)
            assert (b['num_bits'],b['exponent_bits'],b['mantissa_bits'])==(4,2,1)
            counts[scope]+=1
assert counts=={'target':640,'draft':48},counts
(r/'evidence/activation-dtype-gate.json').write_text(json.dumps({'passed':True,'configs':counts,'target':'FP8 E4M3','draft':'BF16','weights':'FP4 E2M1'},indent=2))

assert len(list((r/'evidence/headrow').glob('headrow-rank*.json')))==8
assert len(list((r/'evidence/compact').glob('workspace-rank*.json')))==8
# One isolated synthetic preflight: distinct user-only prompt, no dataset prefix.
payload={'model':model,'messages':[{'role':'user','content':'Adapter preflight 20261116. Summarize these synthetic numbers in several sentences: '+ ' '.join(str(i) for i in range(1024))}],'max_tokens':32,'temperature':0,'stream':False}
request=urllib.request.Request('http://127.0.0.1:8000/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
with urllib.request.urlopen(request,timeout=180) as response: preflight_result=json.load(response)
(r/'evidence/preflight.json').write_text(json.dumps({'purpose':'kernel-branch readiness; excluded from150 timing; synthetic non-dataset prompt','usage':preflight_result.get('usage'),'finish_reasons':[x.get('finish_reason') for x in preflight_result.get('choices',[])]},indent=2))
assert len(list((r/'evidence/compact').glob('compact-branch-rank*.json')))==8, 'Compact path not selected in startup; refusing timed replay'

(r/'evidence/audit-models-ready.json').write_text(json.dumps(models,indent=2))
assert json.loads(pathlib.Path('/root/dsk_opt_goal_20261116/evidence/fastokens-parity.json').read_text())['passed']
assert list((r/'evidence/fastokens').glob('*-backend.json'))
assert list((r/'evidence/fastokens').glob('*-encode*.json')), 'No actual fastokens encode observed'

full_files=list((r/'evidence/full-finalize').glob('branch-rank*.json'))
assert len(full_files)==8 and all(len(json.loads(p.read_text()))==40 for p in full_files),'Full fusion not observed in all 40 layers on all8 ranks'
assert json.loads(pathlib.Path('/root/dsk_opt_goal_20261116/evidence/humming-full-contract-exit.json').read_text())['exit_code']==0

cmd=[sys.executable,str(pathlib.Path('/root/dsk_opt_goal_20261116/scripts/09_replay150.py')),'--save-responses','--plan','/root/bench_dsk_first150_20261116/first150.plan.json.gz','--base-url','http://127.0.0.1:8000/v1','--model',model,'--output',str(r/'results/replay150.jsonl')]
with (r/'logs/replay150.log').open('x') as log: code=subprocess.call(cmd,stdout=log,stderr=subprocess.STDOUT)
(r/'evidence/replay-exit.json').write_text(json.dumps({'exit_code':code,'command':cmd}))
sys.exit(code)
