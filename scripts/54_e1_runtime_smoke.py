import importlib.metadata as md,json,os,pathlib,subprocess,sys,time
root=pathlib.Path('/root/gc_moe_isolated_20261116')
assert sys.prefix==str(root/'.venv'),sys.prefix
modules=['torch','triton','cutlass','gc_moe','torchvision','torchaudio','torchcodec','vllm','vllm._C_stable_libtorch','vllm._moe_C_stable_libtorch','vllm.engine.arg_utils','vllm.models.deepseek_v41','dynamo.runtime','dynamo.vllm','dynamo.frontend']
results=[]
def run(label,cmd):
 start=time.monotonic()
 try:
  p=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
  item={'module':label,'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
 except subprocess.TimeoutExpired:
  item={'module':label,'exit_code':124,'error':'timeout'}
 item['seconds']=time.monotonic()-start;results.append(item);print(json.dumps(item),flush=True)
for mod in modules:
 code=f'import importlib; m=importlib.import_module({mod!r}); print(getattr(m,"__file__",None))'
 if mod=='torch':code+='; assert m.__file__.startswith("/root/gc_moe_isolated_20261116/.venv/")'
 if mod=='vllm':code+='; assert m.__file__.startswith("/root/gc_moe_isolated_20261116/src/vllm-5df63a7/")'
 run(mod,[sys.executable,'-c',code])
if all(x['exit_code']==0 for x in results):
 for name in ['dynamo.vllm','dynamo.frontend']:run(name+' --help',[sys.executable,'-m',name,'--help'])
report={'scope':'CPU import and CLI smoke only, no model loading or GPU correctness claim','python':sys.executable,'pythonpath':os.environ.get('PYTHONPATH'),'results':results,'passed':all(x['exit_code']==0 for x in results),'versions':{d.metadata['Name']:d.version for d in md.distributions()}}
(root/'evidence/runtime-import-smoke.json').write_text(json.dumps(report,indent=2))
sys.exit(0 if report['passed'] else 1)
