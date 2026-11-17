import pathlib,json,shutil
r=pathlib.Path('/root/dsk_opt_goal_20261116');p=r/'src/dsk_humming_full_finalize.py';s=p.read_text()
s=s.replace('import contextvars,os','import contextvars,os,json,pathlib\n_HITS={}')
s=s.replace("ctx['hit']=True", """ctx['hit']=True
        directory=os.environ.get('DSK_FULL_FINALIZE_EVIDENCE_DIR')
        name=getattr(runner,'layer_name',None)
        if directory and name and name not in _HITS:
            import torch.distributed as dist
            _HITS[name]={'rows':out.shape[0],'scale':float(runner.routed_scaling_factor),'shared_fused':True,'async_wait_at_finalizer':ctx['overlap'],'before_EP_reduce':True}
            path=pathlib.Path(directory);path.mkdir(parents=True,exist_ok=True)
            (path/f'branch-rank{dist.get_rank()}.json').write_text(json.dumps(_HITS,indent=2))""")
compile(s,str(p),'exec');p.write_text(s)
old=r/'groups/P-E1-FAST-HUM-ALL-r1';g=r/'groups/P-E1-FAST-HUM-FULL-r1';assert not g.exists()
for d in ['scripts','logs','results','evidence','traces','discovery']:(g/d).mkdir(parents=True)
for p in (old/'scripts').iterdir():
 if p.is_file():
  s=p.read_text().replace(old.name,g.name).replace('p-e1-fast-hum-all-r1','p-e1-fast-hum-full-r1').replace('dsk_combined_worker.Worker','dsk_humming_full_worker.Worker')
  if p.name=='00_runtime_env.sh':s+=f'\nexport DSK_HUMMING_FULL_FINALIZE=1\nexport DSK_FULL_FINALIZE_EVIDENCE_DIR={g}/evidence/full-finalize\n'
  if p.name=='03_replay.py':
   gate="""full_files=list((r/'evidence/full-finalize').glob('branch-rank*.json'))
assert len(full_files)==8 and all(len(json.loads(p.read_text()))==40 for p in full_files),'Full fusion not observed in all 40 layers on all8 ranks'
assert json.loads(pathlib.Path('/root/dsk_opt_goal_20261116/evidence/humming-full-contract-exit.json').read_text())['exit_code']==0
"""
   s=s.replace('cmd=[sys.executable',gate+'\ncmd=[sys.executable')
  (g/'scripts'/p.name).write_text(s)
(g/'evidence/source-snapshot').mkdir()
for name in ['dsk_humming_full_finalize.py','dsk_humming_full_worker.py']:shutil.copy2(r/'src'/name,g/'evidence/source-snapshot'/name)
(g/'evidence/config-delta.json').write_text(json.dumps({'baseline':old.name,'change':'full grouped unpermute + BF16 routed scale + shared add fusion at <=128 rows','unchanged':'E1 runtime, fastokens, all other three optimizations, sampling, data, rate','safety':'no DBO/transforms/SP/DP; shared stream waited at finalizer, shared getter consumed once, allreduce position unchanged; larger batches retain native scale/add and upstream/tiled finalizer'},indent=2))
s=(r/'scripts/85_run_fastokens_group.py').read_text();start=s.index('assert a.group in ');end=s.index('\n',start);s=s[:start]+"assert a.group=='P-E1-FAST-HUM-FULL-r1'"+s[end:]
(r/'scripts/96_run_humming_full.py').write_text(s)
s=(r/'scripts/87_fastokens_sequence_retry.py').read_text().replace("old='P-GC-ALL-w4a8-r4'","old='P-E1-FAST-HUM-ALL-r1'").replace("[('P-FAST-GC-ALL-r2','P-GC-ALL-w4a8-r4'),('P-FAST-HUM-ALL-r1','P-ALL-w4a8-stride-r2')]","[('P-E1-FAST-HUM-FULL-r1','P-E1-FAST-HUM-ALL-r1')]").replace('85_run_fastokens_group.py','96_run_humming_full.py').replace('evidence/fastokens-sequence.json','evidence/humming-full-sequence.json')
(r/'scripts/97_humming_full_sequence.py').write_text(s)
