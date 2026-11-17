import pathlib,subprocess,json,os,sys,time
r=pathlib.Path('/root/gc_moe_isolated_20261116');src=r/'src/vllm-5df63a7/vllm'
paths=[src/'_C_stable_libtorch.abi3.so',src/'_moe_C_stable_libtorch.abi3.so']
paths+=list((src/'third_party/flashmla').glob('*.so'))
report={'scope':'CPU-side dynamic linkage and operator registration only; no GPU kernel/model execution','checks':[]}
for path in paths:
 code='import torch; torch.ops.load_library('+repr(str(path))+'); print(torch.__version__)'
 p=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},timeout=90)
 report['checks'].append({'path':str(path),'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
report['passed']=bool(paths) and all(x['exit_code']==0 for x in report['checks'])
(r/'evidence/compiled-extension-linkage.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));sys.exit(0 if report['passed'] else 1)
