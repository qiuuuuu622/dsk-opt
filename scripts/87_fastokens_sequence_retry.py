import pathlib,subprocess,json,shutil,time,traceback
r=pathlib.Path('/root/dsk_opt_goal_20261116');py=str(r/'.venv-e0/bin/python');state=r/'evidence/fastokens-sequence.json';done=[]
def status(value,**kw):state.write_text(json.dumps({'status':value,'completed':done,'time':time.time(),**kw},indent=2))
old='P-GC-ALL-w4a8-r4'
try:
 for name,base in [('P-FAST-GC-ALL-r2','P-GC-ALL-w4a8-r4'),('P-FAST-HUM-ALL-r1','P-ALL-w4a8-stride-r2')]:
  g=r/'groups'/name;status('running',group=name)
  with (g/'logs/controller.log').open('x') as f:subprocess.run(['python3',str(r/'scripts/85_run_fastokens_group.py'),'--group',name,'--stop-group',old],stdout=f,stderr=subprocess.STDOUT,check=True)
  status('analyzing',group=name)
  with (g/'logs/analysis.log').open('x') as f:
   subprocess.run([py,str(r/'scripts/42_compare_humming.py'),'--group',name,'--baseline',base],stdout=f,stderr=subprocess.STDOUT,check=True)
   subprocess.run([py,str(r/'scripts/33_quality_compare_dsml.py'),'--baseline',str(g/'results/replay150.jsonl'),'--plan','/root/bench_dsk_first150_20261116/first150.plan.json.gz','--output',str(g/'evidence/quality-dsml.json')],stdout=f,stderr=subprocess.STDOUT,check=True)
   d=json.loads((g/'evidence/quality-dsml.json').read_text())['groups']['baseline'];keys=['empty','truncated','malformed_tool_json','repeated_suffix_flag','dsml_unbalanced_flags','dsml_unknown_tools','dsml_duplicate_parameters','dsml_nonstring_json_fallbacks']
   (g/'evidence/quality-gate.json').write_text(json.dumps({'status':'structural_proxy_pass' if d['success']==150 and not any(d[k] for k in keys) else 'needs_review','dsml_calls':d['dsml_calls'],'flags':{k:d[k] for k in keys},'limits':'No business labels; not business accuracy proof'},indent=2))
   subprocess.run([py,str(r/'scripts/41_humming_trace.py'),'--group',name],stdout=f,stderr=subprocess.STDOUT,check=True)
  evidence=json.loads((g/'evidence/profile-diagnostic.json').read_text());counts=[]
  for rank in evidence['profiles']:
   match={'compact':lambda n:'_candidate_wgmma_kernel' in n,'headrow':lambda n:'exchange_kernel' in n}
   if 'GC' in name:match.update(gc_native=lambda n:'SM90GCMoERoleKernel' in n,finalize=lambda n:'_gc_finalize' in n)
   else:match.update(finalize=lambda n:'tiled' in n.lower() and ('finalize' in n.lower() or 'sum' in n.lower()))
   counts.append({'file':rank['file'],'counts':{k:sum(v['count'] for n,v in rank['kernels'].items() if fn(n)) for k,fn in match.items()}})
  (g/'evidence/four-paths.json').write_text(json.dumps({'scope':'post-timing3request trace; manual symbol review required','ranks':counts},indent=2))
  done.append(name);old=name;status('group_completed',group=name)
 status('completed',service_retained=old)
except BaseException as exc:status('failed',error=repr(exc));traceback.print_exc();raise
