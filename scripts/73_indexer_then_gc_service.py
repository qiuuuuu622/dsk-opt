import pathlib,subprocess
r=pathlib.Path('/root/dsk_opt_goal_20261116')
assert (r/'evidence/gc-service-adapter-triton37-exit.txt').read_text().strip()=='0'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=30).strip()
with (r/'logs/indexer-e1-triton37.log').open('x') as log:
 code=subprocess.call(['bash',str(r/'scripts/69_indexer_e1_verify.sh')],stdout=log,stderr=subprocess.STDOUT)
assert code==0,('indexer verification failed',code)
with (r/'groups/P-GC-ALL-w4a8-r2/logs/orchestrator.log').open('x') as log:
 code=subprocess.call(['python3',str(r/'scripts/67_run_gc_combined.py'),'--group','P-GC-ALL-w4a8-r2','--stop-group','P-ALL-w4a8-stride-r2'],stdout=log,stderr=subprocess.STDOUT)
raise SystemExit(code)
