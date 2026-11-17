"""Analyze same-code repeat; no extra full replay or diagnostic load."""
import pathlib,json,subprocess,time,shutil
r=pathlib.Path('/root/dsk_opt_goal_20261116');g=r/'groups/P-ALL-w4a8-stride-r2';py=str(r/'.venv-e0/bin/python')
for _ in range(650):
 state=json.loads((g/'evidence/run-state.json').read_text())
 if state['status']=='completed_service_retained':break
 if state['status']=='failed':raise RuntimeError(state)
 time.sleep(10)
else:raise TimeoutError('Run still incomplete; no restart')
for base,label in [('A1-off-r1','native'),('P-ALL-w4a8-stride-r1','repeat-r1')]:
 subprocess.run([py,str(r/'scripts/42_compare_humming.py'),'--group',g.name,'--baseline',base],check=True)
 for ext in ['json','csv']:shutil.copy2(g/f'results/comparison.{ext}',g/f'results/comparison-{label}.{ext}')
for ext in ['json','csv']:shutil.copy2(g/f'results/comparison-native.{ext}',g/f'results/comparison.{ext}')
subprocess.run([py,str(r/'scripts/33_quality_compare_dsml.py'),'--baseline',str(g/'results/replay150.jsonl'),'--plan','/root/bench_dsk_first150_20261116/first150.plan.json.gz','--output',str(g/'evidence/quality-dsml.json')],check=True)
d=json.loads((g/'evidence/quality-dsml.json').read_text())['groups']['baseline']
keys=['empty','truncated','malformed_tool_json','repeated_suffix_flag','dsml_unbalanced_flags','dsml_unknown_tools','dsml_duplicate_parameters','dsml_nonstring_json_fallbacks']
(g/'evidence/quality-gate.json').write_text(json.dumps({'status':'structural_proxy_pass' if d['success']==150 and not any(d[k] for k in keys) else 'needs_review','dsml_calls':d['dsml_calls'],'flags':{k:d[k] for k in keys},'limits':'No business labels; original stochastic sampling'},indent=2))
print('Repeat analysis complete',flush=True)
