import collections,csv,importlib.util,json,pathlib,math
r=pathlib.Path('/root/dsk_opt_goal_20261116/groups/P-ALL-w4a8-spec6-r1')
p=r/'results/replay150.jsonl'
rows=[json.loads(s) for s in p.read_text().splitlines()]
summary=json.loads(p.with_suffix('.summary.json').read_text())
assert len(rows)==150 and len({x['event_id'] for x in rows})==150
ordered=sorted(rows,key=lambda x:x['index'])
assert [x['index'] for x in ordered]==list(range(1,151))
assert all(a['started_at_sec']<=b['started_at_sec'] for a,b in zip(ordered,ordered[1:]))
spec=importlib.util.spec_from_file_location('replay',pathlib.Path('/root/dsk_opt_goal_20261116/scripts/09_replay150.py')); mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
computed=mod.summarize(rows,summary['elapsed_sec']);assert summary==computed
for x in rows:
 if x['status']=='ok' and x['tpot_ms_per_token'] is not None:
  expected=(x['e2e_ms']-x['ttft_ms'])/(x['completion_tokens']-1)
  assert math.isclose(expected,x['tpot_ms_per_token'])
with (r/'results/percentiles.csv').open('w') as f:
 writer=csv.writer(f);writer.writerow(['metric','n','mean','p50','p90','p95','p99'])
 for metric in ['ttft_ms','tpot_ms_per_token','e2e_ms']:
  d=summary[metric];writer.writerow([metric]+[d[k] for k in ['n','mean','p50','p90','p95','p99']])
report={'validated':True,'requests':len(rows),'success':summary['success'],'failed':summary['failed'],'scheduled_send_end_sec':ordered[-1]['scheduled_at_sec'],'max_dispatch_lag_ms':max(x['dispatch_lag_ms'] for x in rows),'finish_reasons':dict(collections.Counter(y for x in rows for y in x.get('finish_reasons',[]))),'completion_tokens':sum(x.get('completion_tokens') or 0 for x in rows),'all_usage':all(x.get('usage') is not None for x in rows),'send_order_verified':True,'percentiles_recomputed':True}
(r/'evidence/validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));print(json.dumps(summary,indent=2))
