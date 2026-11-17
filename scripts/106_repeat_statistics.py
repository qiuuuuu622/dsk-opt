"""Descriptive two-run spread and paired request bootstrap; not run-level CI."""
import pathlib,json,random,statistics
r=pathlib.Path('/root/dsk_opt_goal_20261116');names=['A1-off-r1','P-E1-FAST-HUM-FULL-r1','P-E1-FAST-HUM-FULL-r2'];data={}
for name in names:
 rows=[json.loads(x) for x in (r/'groups'/name/'results/replay150.jsonl').read_text().splitlines()];assert len(rows)==150 and all(x['status']=='ok' for x in rows);data[name]={x['event_id']:x for x in rows}
ids=list(data[names[0]]);assert all(set(data[n])==set(ids) for n in names)
def quantile(a,q):
 a=sorted(a);i=(len(a)-1)*q/100;lo=int(i);hi=min(lo+1,len(a)-1);return a[lo]+(a[hi]-a[lo])*(i-lo)
rows=[]
for key in ['ttft_ms','tpot_ms_per_token','e2e_ms']:
 arrays=[[data[name][i][key] for i in ids] for name in names];assert all(all(x is not None for x in a) for a in arrays)
 for q in [95,99]:
  observed=[quantile(a,q) for a in arrays];rng=random.Random(20261116);delta=[]
  for _ in range(3000):
   ii=[rng.randrange(150) for _ in range(150)];qs=[quantile([a[i] for i in ii],q) for a in arrays];delta.append(statistics.mean(qs[1:])-qs[0])
  rows.append({'metric':key,'quantile':q,'baseline':observed[0],'rounds':observed[1:],'median_across_two_rounds':statistics.median(observed[1:]),'range_across_two_rounds':[min(observed[1:]),max(observed[1:])],'paired_request_bootstrap_mean_quantile_difference_95pct':[quantile(delta,2.5),quantile(delta,97.5)]})
report={'baseline':names[0],'candidates':names[1:],'seed':20261116,'draws':3000,'scope':'Resample matched request IDs jointly across all three runs, average candidate quantiles minus baseline. This is NOT a CI over independent runs, NOT a pure mechanism causal estimate. One baseline and stochastic generated lengths, E0/E1 runtime differences limit attribution. Two candidate runs do not establish P99 stability.','rows':rows}
(r/'results/final-repeat-statistics.json').write_text(json.dumps(report,indent=2));print(json.dumps(rows,indent=2))
