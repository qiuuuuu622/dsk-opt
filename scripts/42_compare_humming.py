import argparse,json,pathlib,csv
p=argparse.ArgumentParser();p.add_argument('--group',required=True);p.add_argument('--baseline',default='A1-off-r1');a=p.parse_args()
r=pathlib.Path('/root/dsk_opt_goal_20261116');g=r/'groups'/a.group;b=r/'groups'/a.baseline
s=json.loads((g/'results/replay150.summary.json').read_text());base=json.loads((b/'results/replay150.summary.json').read_text())
assert s['finished']==150 and base['finished']==150
raw=[json.loads(x) for x in (g/'results/replay150.jsonl').read_text().splitlines()];br=[json.loads(x) for x in (b/'results/replay150.jsonl').read_text().splitlines()]
assert {(x['index'],x['event_id'],x['scheduled_at_sec']) for x in raw}=={(x['index'],x['event_id'],x['scheduled_at_sec']) for x in br}
rows=[]
for metric in ['ttft_ms','tpot_ms_per_token','e2e_ms']:
 for q in ['mean','p50','p90','p95','p99']:
  x=base[metric][q];y=s[metric][q];rows.append({'metric':metric,'stat':q,'baseline':x,'candidate':y,'change_percent':100*(y/x-1)})
report={'group':a.group,'baseline':a.baseline,'scope':'single-run quick screen against existing baseline; not statistical significance','success':s['success'],'failed':s['failed'],'latency':rows,'output_tokens':{'baseline':sum(x.get('completion_tokens') or 0 for x in br),'candidate':sum(x.get('completion_tokens') or 0 for x in raw)},'output_tokens_per_sec':{'baseline':base['reported_output_tokens_per_sec'],'candidate':s['reported_output_tokens_per_sec']},'elapsed_seconds':{'baseline':base['elapsed_sec'],'candidate':s['elapsed_sec']},'caveats':['Original temperature0.8 sampling gives varying generated lengths',('Profiler configured but inactive during both timed replays' if '--profiler-config' in (b/'scripts/01_worker.sh').read_text() else 'Profiler configured but disabled during candidate timed replay; baseline lacked profiler configuration'),'GC compile suspended during candidate timed replay']}
(g/'results/comparison.json').write_text(json.dumps(report,indent=2))
with (g/'results/comparison.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps({**{k:v for k,v in report.items() if k not in ['latency','caveats']},'p95':[x for x in rows if x['stat']=='p95']},indent=2))
