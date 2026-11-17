import pathlib,json,os,signal,time,subprocess,shutil
r=pathlib.Path('/root/dsk_opt_goal_20261116'); old=r/'groups/P-GC-ALL-w4a8-r2'; new=r/'groups/P-GC-ALL-w4a8-r3'
assert not new.exists()
for pid,needle in [(437678,'67_run_gc_combined.py'),(441653,'74_finalize_gc_combined.py')]:
 p=pathlib.Path(f'/proc/{pid}/cmdline')
 if p.exists():
  assert needle in p.read_text();os.kill(pid,signal.SIGTERM)
pids=json.loads((old/'evidence/service-pids.json').read_text())
for key,pid in pids.items():
 try:
  assert os.getpgid(pid)==pid
  os.killpg(pid,signal.SIGTERM)
 except ProcessLookupError:pass
time.sleep(15)
for pid in pids.values():
 try:os.killpg(pid,signal.SIGKILL)
 except ProcessLookupError:pass
rows=[json.loads(x) for x in (old/'results/replay150.jsonl').read_text().splitlines()]
(old/'evidence/run-state.json').write_text(json.dumps({'status':'failed','reason':'Isolated Dynamo Rust frontend rejects dataset assistant content with HTTP400; preserve original data and use baseline frontend environment in r3','finished':len(rows),'success':sum(x['status']=='ok' for x in rows),'pids':pids},indent=2))
for d in ['scripts','logs','results','evidence','traces','discovery']:(new/d).mkdir(parents=True)
for p in (old/'scripts').iterdir():
 if p.is_file():(new/'scripts'/p.name).write_text(p.read_text().replace('P-GC-ALL-w4a8-r2','P-GC-ALL-w4a8-r3').replace('dsk-gc-all-w4a8-r2','dsk-gc-all-w4a8-r3'))
# Keep exact baseline frontend Python and binary runtime, isolated GPU worker unchanged.
(new/'scripts/02_frontend.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
export DYN_DISCOVERY_BACKEND=file DYN_REQUEST_PLANE=tcp DYN_EVENT_PLANE=zmq
export DYN_FILE_KV=/root/dsk_opt_goal_20261116/groups/P-GC-ALL-w4a8-r3/discovery
export DYN_NAMESPACE=dsk-gc-all-w4a8-r3 PYTHONUNBUFFERED=1
unset PYTHONPATH
exec /root/dsk_opt_goal_20261116/.venv-e0/bin/python -m dynamo.frontend --dyn-chat-processor vllm --http-host 127.0.0.1 --http-port 8000 --discovery-backend file --request-plane tcp --event-plane zmq
''')
(new/'evidence/frontend-alignment.json').write_text(json.dumps({'frontend':'Exact E0 baseline frontend Python/runtime','worker':'E1 GC Torch2.11 Triton3.7.1 unchanged','reason':'E1 wheel Rust HTTP schema rejects some original assistant content; no dataset rewrite','previous_run':'P-GC-ALL-w4a8-r2'},indent=2))
s=(r/'scripts/67_run_gc_combined.py').read_text().replace("'P-GC-ALL-w4a8-r2')","'P-GC-ALL-w4a8-r2','P-GC-ALL-w4a8-r3')")
a=s.index('def stop_old():');b=s.index('def launch(',a)
s=s[:a]+'''def stop_old():
 out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=30)
 assert not out.strip(),('GPU processes still present',out)
'''+s[b:]
(r/'scripts/76_run_gc_aligned_frontend.py').write_text(s)
w=(r/'scripts/74_finalize_gc_combined.py').read_text().replace('P-GC-ALL-w4a8-r2','P-GC-ALL-w4a8-r3')
(r/'scripts/77_finalize_gc_combined_r3.py').write_text(w)
with (r/'logs/gc-combined-r3-controller.log').open('x') as f:
 p=subprocess.Popen(['python3',str(r/'scripts/76_run_gc_aligned_frontend.py'),'--group',new.name,'--stop-group',old.name],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print('controller',p.pid,flush=True)
while not (new/'evidence/run-state.json').exists():time.sleep(1)
with (r/'logs/gc-combined-finalize-r3.log').open('x') as f:
 p=subprocess.Popen(['python3',str(r/'scripts/77_finalize_gc_combined_r3.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print('finalizer',p.pid,flush=True)
