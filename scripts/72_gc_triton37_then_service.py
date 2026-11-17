import pathlib,time,subprocess,json
r=pathlib.Path('/root/dsk_opt_goal_20261116');pid=436425;p=pathlib.Path(f'/proc/{pid}')
if p.exists():
 assert b'71_gc_triton37_validation.sh' in (p/'cmdline').read_bytes()
 start=(p/'stat').read_text().split(') ',1)[1].split()[19]
 while p.exists():
  try:
   stat=(p/'stat').read_text().split(') ',1)[1].split()
   if stat[19]!=start or stat[0]=='Z':break
  except FileNotFoundError:break
  time.sleep(5)
for name in ['gc-service-adapter-triton37-exit.txt','indexer-strided-e1-exit.txt']:
 assert (r/'evidence'/name).read_text().strip()=='0',name
with (r/'groups/P-GC-ALL-w4a8-r2/logs/orchestrator.log').open('x') as log:
 code=subprocess.call(['python3',str(r/'scripts/67_run_gc_combined.py'),'--group','P-GC-ALL-w4a8-r2','--stop-group','P-ALL-w4a8-stride-r2'],stdout=log,stderr=subprocess.STDOUT)
raise SystemExit(code)
