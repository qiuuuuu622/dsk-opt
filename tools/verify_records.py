"""Offline validation of exported metrics and provenance; no GPU required."""
import hashlib,json,math,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
E=ROOT/'experiments/20261116'
def quantile(v,q):
 v=sorted(v);p=(len(v)-1)*q;i=int(p);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(p-i)
checked=0
for p in sorted((E/'groups').glob('*/results/replay150.summary.json')):
 s=json.loads(p.read_text());raw=p.with_name('replay150.jsonl')
 if not raw.exists() or s.get('success')!=150 or s.get('failed')!=0:continue
 rows=[json.loads(x) for x in raw.read_text().splitlines() if x.strip()]
 ok=[x for x in rows if x['status']=='ok'];assert len(ok)==150,p
 for k in ('ttft_ms','tpot_ms_per_token','e2e_ms'):
  vals=[r[k] for r in ok if isinstance(r.get(k),(float,int))]
  for name,q in [('p50',.5),('p90',.9),('p95',.95),('p99',.99)]:
   assert math.isclose(quantile(vals,q),s[k][name],rel_tol=1e-9,abs_tol=1e-7),(p,k,name)
 checked+=1
manifest=json.loads((E/'EXPORT-MANIFEST.json').read_text())
for r in manifest:
 assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['published_sha256'],r['path']
print(f'PASS: {checked} complete groups, 12 quantiles/group; {len(manifest)} file hashes')
