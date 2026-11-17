import pathlib,json,shutil
r=pathlib.Path('/root/dsk_opt_goal_20261116')
for name,base in [('P-FAST-GC-ALL-r1','P-GC-ALL-w4a8-r4'),('P-FAST-HUM-ALL-r1','P-ALL-w4a8-stride-r2')]:
 g=r/'groups'/name;b=r/'groups'/base
 assert not g.exists()
 for d in ['scripts','logs','results','evidence','traces','discovery']:(g/d).mkdir(parents=True)
 for p in (b/'scripts').iterdir():
  if p.is_file():
   s=p.read_text().replace(base,name)
   import re
   s=re.sub(r'export DYN_NAMESPACE=\S+',f'export DYN_NAMESPACE={name.lower()}',s)
   s=s.replace('-m dynamo.vllm','/root/dsk_opt_goal_20261116/src/dsk_fastokens_entry.py dynamo.vllm')
   s=s.replace('-m dynamo.frontend','/root/dsk_opt_goal_20261116/src/dsk_fastokens_entry.py dynamo.frontend')
   if p.name in ['00_runtime_env.sh','02_frontend.sh']:
    opts=f'export VLLM_USE_FASTOKENS=1\nexport DYN_TOKENIZER=fastokens DYN_TOKENIZER_FALLBACK=false\nexport DSK_FASTOKENS_EVIDENCE_DIR={g}/evidence/fastokens\n'
    if p.name=='02_frontend.sh':s=s.replace('exec /root/',opts+'exec /root/')
    else:s+='\n'+opts
   if p.name=='03_replay.py':
    marker="cmd=[sys.executable"
    s=s.replace(marker,"assert json.loads(pathlib.Path('/root/dsk_opt_goal_20261116/evidence/fastokens-parity.json').read_text())['passed']\nassert list((r/'evidence/fastokens').glob('*-backend.json'))\nassert list((r/'evidence/fastokens').glob('*-encode*.json')), 'No actual fastokens encode observed'\n\n"+marker)
   (g/'scripts'/p.name).write_text(s)
 (g/'evidence/config-delta.json').write_text(json.dumps({'baseline':base,'fastokens':'0.3.6','VLLM_USE_FASTOKENS':'1','DYN_TOKENIZER':'fastokens','DYN_TOKENIZER_FALLBACK':'false','worker_and_frontend':True,'other_parameters':'unchanged from own baseline','Humming_finalize':'existing tiled finalize only; GC has scale/shared fusion'},indent=2))
 (g/'evidence/source-snapshot').mkdir()
 shutil.copy2(r/'src/dsk_fastokens_entry.py',g/'evidence/source-snapshot/dsk_fastokens_entry.py')
 if 'GC' in name:shutil.copy2(r/'src/dsk_gc_adapter.py',g/'evidence/source-snapshot/dsk_gc_adapter.py')
s=(r/'scripts/67_run_gc_combined.py').read_text()
s=s.replace("assert a.group in ('P-GC-ALL-w4a8-r1','P-GC-ALL-w4a8-r2')","assert a.group in ('P-FAST-GC-ALL-r1','P-FAST-HUM-ALL-r1')")
s=s.replace("assert (ROOT/'evidence/gc-service-adapter-triton37-exit.txt').read_text().strip()=='0'","assert json.loads((ROOT/'evidence/fastokens-parity.json').read_text())['passed']\n assert (ROOT/'evidence/gc-stable-graph-exit.txt').read_text().strip()=='0'")
(r/'scripts/85_run_fastokens_group.py').write_text(s)
