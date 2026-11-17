import pathlib,json,subprocess,time,shutil
r=pathlib.Path('/root/dsk_opt_goal_20261116');old=r/'groups/P-GC-ALL-w4a8-r3';new=r/'groups/P-GC-ALL-w4a8-r4'
while not (r/'evidence/gc-stable-graph-exit.txt').exists():time.sleep(5)
assert (r/'evidence/gc-stable-graph-exit.txt').read_text().strip()=='0'
assert not new.exists()
for d in ['scripts','logs','results','evidence','traces','discovery']:(new/d).mkdir(parents=True)
for p in (old/'scripts').iterdir():
 if p.is_file():(new/'scripts'/p.name).write_text(p.read_text().replace('P-GC-ALL-w4a8-r3','P-GC-ALL-w4a8-r4').replace('dsk-gc-all-w4a8-r3','dsk-gc-all-w4a8-r4'))
shutil.copy2(old/'evidence/frontend-alignment.json',new/'evidence/frontend-alignment.json')
(new/'evidence/source-snapshot').mkdir();shutil.copy2(r/'src/dsk_gc_adapter.py',new/'evidence/source-snapshot/dsk_gc_adapter.py')
(new/'evidence/graph-cache-fix.json').write_text(json.dumps({'fixed_output_buffer':True,'per_rank_rows':'next_power_of_two(ceil(M/8)), zero route weights for padding','native_GC_kernel_unchanged':True,'correctness':'160 bitwise native GC comparisons + 320 graph checks passed on 8 ranks','previous_failure':'r3 CUDA child graph instantiate OOM; dynamic pointer/shape cache growth suspected'},indent=2))
s=(r/'scripts/76_run_gc_aligned_frontend.py').read_text().replace("'P-GC-ALL-w4a8-r3')","'P-GC-ALL-w4a8-r3','P-GC-ALL-w4a8-r4')")
a=s.index('def stop_old():');b=s.index('def launch(',a)
s=s[:a]+'''def stop_old():
 for _ in range(30):
  try:
   out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=10)
   if not out.strip():return
  except subprocess.TimeoutExpired:pass
  time.sleep(5)
 raise RuntimeError('GPU processes still present')
'''+s[b:]
(r/'scripts/81_run_gc_stable_group.py').write_text(s)
w=(r/'scripts/77_finalize_gc_combined_r3.py').read_text().replace('P-GC-ALL-w4a8-r3','P-GC-ALL-w4a8-r4')
(r/'scripts/82_finalize_gc_r4.py').write_text(w)
with (r/'logs/gc-combined-r4-controller.log').open('x') as f:p=subprocess.Popen(['python3',str(r/'scripts/81_run_gc_stable_group.py'),'--group',new.name,'--stop-group',old.name],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print('controller',p.pid,flush=True)
while not (new/'evidence/run-state.json').exists():time.sleep(1)
with (r/'logs/gc-combined-finalize-r4.log').open('x') as f:p=subprocess.Popen(['python3',str(r/'scripts/82_finalize_gc_r4.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print('finalizer',p.pid,flush=True)
