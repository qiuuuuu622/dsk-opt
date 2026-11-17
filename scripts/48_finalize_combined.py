"""Finalize an existing combined replay, then collect separate kernel evidence."""
import argparse,json,pathlib,subprocess,time,shutil
p=argparse.ArgumentParser();p.add_argument('--group',required=True);a=p.parse_args()
r=pathlib.Path('/root/dsk_opt_goal_20261116');g=r/'groups'/a.group;py=str(r/'.venv-e0/bin/python')
for _ in range(180):
 state=json.loads((g/'evidence/run-state.json').read_text())
 if state['status']=='completed_service_retained':break
 if state['status']=='failed':raise RuntimeError(state)
 time.sleep(10)
else:raise TimeoutError('Existing run not complete; no restart performed')
for base,label in [('A1-off-r1','native'),('P-H3-a8-finalize-r1','h3'),('P-ALL-w4a8-r1','without-compact-proof')]:
 subprocess.run([py,str(r/'scripts/42_compare_humming.py'),'--group',g.name,'--baseline',base],check=True)
 for ext in ['json','csv']:shutil.copy2(g/f'results/comparison.{ext}',g/f'results/comparison-{label}.{ext}')
for ext in ['json','csv']:shutil.copy2(g/f'results/comparison-native.{ext}',g/f'results/comparison.{ext}')
subprocess.run([py,str(r/'scripts/33_quality_compare_dsml.py'),'--baseline',str(g/'results/replay150.jsonl'),'--plan','/root/bench_dsk_first150_20261116/first150.plan.json.gz','--output',str(g/'evidence/quality-dsml.json')],check=True)
d=json.loads((g/'evidence/quality-dsml.json').read_text())['groups']['baseline']
keys=['empty','truncated','malformed_tool_json','repeated_suffix_flag','dsml_unbalanced_flags','dsml_unknown_tools','dsml_duplicate_parameters','dsml_nonstring_json_fallbacks']
(g/'evidence/quality-gate.json').write_text(json.dumps({'status':'structural_proxy_pass' if d['success']==150 and not any(d[k] for k in keys) else 'needs_review','requests':d['requests'],'dsml_calls':d['dsml_calls'],'flags':{k:d[k] for k in keys},'limits':'No business labels; not semantic accuracy proof; original stochastic sampling'},indent=2))
subprocess.run([py,str(r/'scripts/41_humming_trace.py'),'--group',g.name],check=True)
subprocess.run([py,str(r/'scripts/45_combined_path_report.py'),'--group',g.name],check=True)
proof=json.loads((g/'evidence/combined-path-proof.json').read_text())
print(json.dumps({'group':g.name,'all_paths_observed':proof['all_8_ranks_all_paths_observed'],'quality':json.loads((g/'evidence/quality-gate.json').read_text())['status']}),flush=True)
