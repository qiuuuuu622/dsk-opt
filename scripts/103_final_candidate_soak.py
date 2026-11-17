"""Long stability only: whole first150 batches,0.45QPS within each,drain between.
Warm prefix caches retained; these are NOT additional cold A/B measurements.
30min gate then2h gate; each minimum duration finishes its current150 batch.
"""
import pathlib,json,subprocess,time,os,re,traceback
r=pathlib.Path('/root/dsk_opt_goal_20261116');g=r/'groups/P-E1-FAST-HUM-FULL-r2';root=r/'stability/full-final-candidate';py=str(r/'.venv-e0/bin/python')
assert not root.exists(),'Never duplicate or overwrite a stability run'
assert json.loads((r/'evidence/humming-full-repeat-sequence.json').read_text())['status']=='completed'
for name in ['P-E1-FAST-HUM-FULL-r1','P-E1-FAST-HUM-FULL-r2']:
 assert json.loads((r/'groups'/name/'evidence/quality-gate.json').read_text())['status']=='structural_proxy_pass'
pids=json.loads((g/'evidence/service-pids.json').read_text());worker=pids['worker'];front=pids['frontend']
for pid in [worker,front]:
 env=dict(x.split('=',1) for x in pathlib.Path(f'/proc/{pid}/environ').read_bytes().decode().split('\0') if '=' in x)
 assert env.get('DYN_NAMESPACE')=='p-e1-fast-hum-full-r2'
root.mkdir(parents=True);(root/'config.json').write_text(json.dumps({'source_group':g.name,'pids':pids,'plan':'originalfirst150','batch_requests':150,'intrabatch_qps':0.45,'cache':'warm, no reset between batches','phases_min_seconds':[1800,7200],'stop':'finish each150 then assess; errors halt subsequent phases','not_cold_performance_comparison':True},indent=2))
state=root/'state.json';completed=[];proc=None;logpath=g/'logs/worker.log';log_offset=logpath.stat().st_size
monitor=(root/'monitor.jsonl').open('x')
def status(value,**kw):state.write_text(json.dumps({'status':value,'controller_pid':os.getpid(),'service_pids':pids,'completed_phases':completed,'time':time.time(),**kw},indent=2))
def check_live():
 global log_offset
 for pid in [worker,front]:
  p=pathlib.Path(f'/proc/{pid}/stat');assert p.exists() and p.read_text().split(') ',1)[1][0]!='Z',f'Service PID{pid} terminated'
 snapshot={'time':time.time()}
 try:snapshot['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu,temperature.gpu,power.draw','--format=csv,noheader,nounits'],text=True,timeout=10).strip().splitlines()
 except subprocess.TimeoutExpired:snapshot['gpu_query']='timeout; not considered service exit'
 with logpath.open() as f:f.seek(log_offset);chunk=f.read();log_offset=f.tell()
 bad=[x for x in chunk.splitlines() if re.search(r'\bERROR\b|Traceback|illegal memory access|CUDA_ERROR_OUT_OF_MEMORY',x)]
 snapshot['error_lines']=bad[:10];monitor.write(json.dumps(snapshot)+'\n');monitor.flush()
 assert not bad,'New service error during stability; inspect monitor'
try:
 for phase,minimum in [('30min',1800),('2h',7200)]:
  phase_root=root/phase;phase_root.mkdir();start=time.monotonic();batch=0;total=0
  while time.monotonic()-start<minimum:
   batch+=1;b=phase_root/f'batch{batch:03d}';b.mkdir();output=b/'replay150.jsonl'
   status('running',phase=phase,batch=batch,finished_requests=total,phase_elapsed=time.monotonic()-start)
   cmd=[py,str(r/'scripts/09_replay150.py'),'--save-responses','--plan','/root/bench_dsk_first150_20261116/first150.plan.json.gz','--base-url','http://127.0.0.1:8000/v1','--model','sf-finance-dsk41-dynamo-vllm-alyeast-prod','--output',str(output)]
   with (b/'replay.log').open('x') as log:
    proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);deadline=time.monotonic()+1200
    while proc.poll() is None:
     check_live()
     raw=output.read_text() if output.exists() else ''
     lines=raw.splitlines()
     if raw and not raw.endswith('\n'):lines=lines[:-1]
     rows=[json.loads(x) for x in lines]
     assert not any(x['status']!='ok' for x in rows),'Failed request in stability batch'
     status('running',phase=phase,batch=batch,replay_pid=proc.pid,finished_requests=total+len(rows),phase_elapsed=time.monotonic()-start)
     assert time.monotonic()<deadline,'Batch exceeded20minutes; stop and inspect'
     time.sleep(30)
   assert proc.returncode==0,('Replay exit',proc.returncode)
   s=json.loads(output.with_suffix('.summary.json').read_text());assert s['success']==150 and s['failed']==0
   with (b/'quality.log').open('x') as log:subprocess.run([py,str(r/'scripts/33_quality_compare_dsml.py'),'--baseline',str(output),'--plan','/root/bench_dsk_first150_20261116/first150.plan.json.gz','--output',str(b/'quality.json')],stdout=log,stderr=subprocess.STDOUT,check=True)
   q=json.loads((b/'quality.json').read_text())['groups']['baseline'];flags=['empty','truncated','malformed_tool_json','repeated_suffix_flag','dsml_unbalanced_flags','dsml_unknown_tools','dsml_duplicate_parameters','dsml_nonstring_json_fallbacks']
   assert not any(q[x] for x in flags),('Quality proxy requires review',{x:q[x] for x in flags})
   total+=150
  record={'phase':phase,'elapsed_sec':time.monotonic()-start,'batches':batch,'requests':total,'status':'completed'};completed.append(record);(phase_root/'summary.json').write_text(json.dumps(record,indent=2));status('phase_completed',phase=phase)
 status('completed')
except BaseException as exc:
 if proc is not None and proc.poll() is None:
  import signal
  os.killpg(proc.pid,signal.SIGTERM)
 status('failed',error=repr(exc));traceback.print_exc();raise
finally:monitor.close()
