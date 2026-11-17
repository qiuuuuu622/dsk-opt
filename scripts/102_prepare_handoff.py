import pathlib,json,hashlib,subprocess
r=pathlib.Path('/root/dsk_opt_goal_20261116');d=r/'handoff/native-off';d.mkdir(parents=True,exist_ok=True)
base=r/'groups/A1-off-r1/scripts'
for name in ['00_runtime_env.sh','01_worker.sh','02_frontend.sh']:
 s=(base/name).read_text().replace('/groups/A1-off-r1','/handoff/native-off').replace('dsk-a1-off-r1','dsk-restore-native-final')
 if name=='00_runtime_env.sh':s+='\nexport DSK_HUMMING_A8=0 DSK_HUMMING_FINALIZE=0 DSK_HUMMING_FULL_FINALIZE=0\nexport DSK_COMPACT_INDEXER=0 DSK_HEADROW=0 DSK_GC_MOE=0 DSK_GC_FINALIZE=0\nexport VLLM_USE_FASTOKENS=0 DYN_TOKENIZER=default DYN_TOKENIZER_FALLBACK=true\n'
 (d/name).write_text(s)
 subprocess.run(['bash','-n',str(d/name)],check=True)
files=json.loads((r/'evidence/baseline-freeze.json').read_text())['files']
actual={name:hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest() for name in files}
assert actual==files,'Baseline source was modified'
(r/'evidence/final-native-source-audit.json').write_text(json.dumps({'unchanged':True,'files':actual,'scope':'three frozen integration entry files; not a whole-site-packages hash'},indent=2))
paths=list((r/'src').glob('dsk_*.py'))+list((r/'src/vllm_dsv41_moe').glob('*.py'))+list((r/'build/headrow').glob('*.so'))
(r/'handoff/source-manifest.json').write_text(json.dumps({'files':{str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'runtime_requirements':'See per-group launch scripts and frozen environment evidence; E1 CUDA extensions must not be copied into E0','native_restore':'native-off/01_worker.sh and02_frontend.sh are prepared only; actual restart and probe pending'},indent=2))
