import json,subprocess,pathlib,sys,datetime
r=pathlib.Path('/root/dsk_opt_goal_20261116')
base=['timeout','--signal=TERM','--kill-after=30s','1800s',sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=8',str(r/'scripts/18_quant_utils_harness.py')]
tests=[]
for rep in range(1,4):
 tests.append((f'routed-graph-1000-r{rep}','tests/test_sm90_gc_moe_readiness.py',['--num-processes','8','--mode','graph','--steps','1000','--num-tokens','64']))
tests.append(('shared-graph-1000','tests/test_sm90_gc_moe_se_readiness.py',['--num-processes','8','--mode','graph','--steps','1000','--num-tokens','64']))
for mode in ['routed','shared']:
 script='tests/test_sm90_gc_moe.py' if mode=='routed' else 'tests/test_sm90_gc_moe_se.py'
 opts=['--gc-only','--hidden','5120','--intermediate-hidden','2304','--num-experts','384','--num-topk','6','--activation-clamp','10','--gc-replays','5']
 if mode=='shared': opts+=['--num-shared-experts','1','--shared-weight-gran-k','32']
 tests.append((mode+'-accuracy',script,opts))
 tests.append((mode+'-edges',script,opts+['--case-category','asymmetric','--per-rank-tokens-list','0,0,0,0,0,0,0,0;0,1,0,1,0,1,0,1;15,16,17,31,32,33,255,256;256,0,1,17,0,32,128,255']))
tests=[(name+'-auditfix',script,args) for name,script,args in tests if name in ['shared-accuracy','shared-edges']]
with (r/'results/gc-shared-auditfix-events.jsonl').open('x') as events:
 for name,script,args in tests:
  if 'readiness' in script: args+=['--output-dir',str(r/'results'/name)]
  else: args+=['--output-json',str(r/'results'/f'{name}.json')]
  cmd=base+[script]+args
  events.write(json.dumps({'name':name,'event':'start','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':cmd})+'\n'); events.flush()
  with (r/'logs'/f'{name}.log').open('x') as log: code=subprocess.call(cmd,stdout=log,stderr=subprocess.STDOUT)
  events.write(json.dumps({'name':name,'event':'end','code':code,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})+'\n');events.flush()
  print(name,code,flush=True)
  if code: sys.exit(code)
