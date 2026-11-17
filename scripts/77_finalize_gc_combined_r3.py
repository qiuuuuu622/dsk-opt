"""One GC combined replay: analyze results, then collect untimed path evidence."""
import pathlib,json,subprocess,time,shutil,os
r=pathlib.Path('/root/dsk_opt_goal_20261116');g=r/'groups/P-GC-ALL-w4a8-r3';py=str(r/'.venv-e0/bin/python')
while True:
 state=json.loads((g/'evidence/run-state.json').read_text())
 if state['status']=='completed_service_retained':break
 if state['status']=='failed':raise RuntimeError(state)
 pid=state.get('pids',{}).get('worker')
 if pid and not pathlib.Path(f'/proc/{pid}').exists():raise RuntimeError('Worker missing before completion; inspect existing run')
 time.sleep(10)
for base,label in [('A1-off-r1','native-E0'),('P-ALL-w4a8-stride-r2','humming-combined-E0')]:
 subprocess.run([py,str(r/'scripts/42_compare_humming.py'),'--group',g.name,'--baseline',base],check=True)
 p=g/'results/comparison.json';data=json.loads(p.read_text())
 data['scope']='Cross-environment single-run whole-configuration comparison; not isolated causal GC gain'
 data['caveats']=['Original temperature0.8 sampling changes generated lengths','Profiler inactive during timed replay; subsequent trace requests excluded','E1 Torch2.11/CUTLASS4.8.1/NCCL2.28.9 differs from E0 Torch2.13/CUTLASS4.8.0/NCCL2.30.7; both now Triton3.7.1','GC FP8 K128 and Humming per-token FP8 differ in activation quantization','Same-environment Humming control not executed in this direct-combination run requested by user']
 p.write_text(json.dumps(data,indent=2))
 for ext in ['json','csv']:shutil.copy2(g/f'results/comparison.{ext}',g/f'results/comparison-{label}.{ext}')
subprocess.run([py,str(r/'scripts/33_quality_compare_dsml.py'),'--baseline',str(g/'results/replay150.jsonl'),'--plan','/root/bench_dsk_first150_20261116/first150.plan.json.gz','--output',str(g/'evidence/quality-dsml.json')],check=True)
d=json.loads((g/'evidence/quality-dsml.json').read_text())['groups']['baseline']
keys=['empty','truncated','malformed_tool_json','repeated_suffix_flag','dsml_unbalanced_flags','dsml_unknown_tools','dsml_duplicate_parameters','dsml_nonstring_json_fallbacks']
(g/'evidence/quality-gate.json').write_text(json.dumps({'status':'structural_proxy_pass' if d['success']==150 and not any(d[k] for k in keys) else 'needs_review','dsml_calls':d['dsml_calls'],'flags':{k:d[k] for k in keys},'limits':'No business labels; stochastic sampling; not business accuracy proof'},indent=2))
subprocess.run([py,str(r/'scripts/41_humming_trace.py'),'--group',g.name],check=True)
p=json.loads((g/'evidence/profile-diagnostic.json').read_text());rows=[]
for profile in p['profiles']:
 kernels=profile['kernels']
 tests={'GC_input_staging':lambda n:'_stage_gc' in n,'GC_output_scale_shared_fusion':lambda n:'_gc_finalize' in n,'compact_scores':lambda n:'_candidate_wgmma_kernel' in n,'candidate_publish':lambda n:'publish_candidate_blocks' in n,'headrow_exchange':lambda n:'exchange_kernel' in n}
 rows.append({'file':profile['file'],'counts':{label:sum(v['count'] for n,v in kernels.items() if fn(n)) for label,fn in tests.items()},'possible_native_GC_symbols':[n for n in kernels if 'gcmoe' in n.lower() or 'gc_moe' in n.lower() or 'sm90gc' in n.lower()]})
(g/'evidence/gc-combined-path-evidence.json').write_text(json.dumps({'scope':'post-timing three-request diagnostic; not whole150 hit rate','ranks':rows,'native_GC_symbol_review':'Required before asserting native GC kernel-hit counts'},indent=2))
print('GC combined analysis and post-timing trace collection complete',flush=True)
