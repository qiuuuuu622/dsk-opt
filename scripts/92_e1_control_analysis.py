import pathlib,json,subprocess,shutil,importlib.metadata as md
r=pathlib.Path('/root/dsk_opt_goal_20261116');g=r/'groups/P-E1-FAST-HUM-ALL-r1';p=g/'evidence'
assert json.loads((r/'evidence/e1-humming-sequence.json').read_text())['status']=='completed'
for suffix in ['json','csv']:shutil.copy2(g/f'results/comparison.{suffix}',g/f'results/comparison-E1-GC.{suffix}')
subprocess.run([str(r/'.venv-e0/bin/python'),str(r/'scripts/42_compare_humming.py'),'--group',g.name,'--baseline','P-FAST-HUM-ALL-r1'],check=True,stdout=subprocess.DEVNULL)
for suffix in ['json','csv']:shutil.copy2(g/f'results/comparison.{suffix}',g/f'results/comparison-E0-Humming.{suffix}')
a=json.loads((p/'profile-diagnostic.json').read_text());rows=[]
for prof in a['profiles']:
 patterns={'compact':lambda n:'_candidate_wgmma_kernel' in n,'headrow':lambda n:'exchange_kernel' in n,'moe_w4a8':lambda n:n.startswith('void humming') and 'FloatingPointType<8u, 4u, 3u>, FloatingPointType<4u, 2u, 1u>' in n,'finalize':lambda n:n in ('_grouped','_indexed')}
 counts={k:sum(v['count'] for n,v in prof['kernels'].items() if fn(n)) for k,fn in patterns.items()};assert all(v>0 for v in counts.values());rows.append({'file':prof['file'],'counts':counts})
assert len(rows)==8
(p/'four-paths.json').write_text(json.dumps({'scope':'post-timing3request trace, not whole150 hit rate','all_four_on_eight_ranks':True,'ranks':rows},indent=2))
(p/'e1-effective-package-versions.json').write_text(json.dumps({n:md.version(n) for n in ['torch','triton','humming-kernels','nvidia-nccl-cu13','fastokens']},indent=2))
report={}
for group in [g.name,'P-FAST-GC-ALL-r2','P-FAST-HUM-ALL-r1']:
 raw=[json.loads(x) for x in (r/'groups'/group/'results/replay150.jsonl').read_text().splitlines()]
 report[group]={'summary':json.loads((r/'groups'/group/'results/replay150.summary.json').read_text()),'completion_tokens':sum(x['completion_tokens'] for x in raw)}
(g/'results/same-environment-summary.json').write_text(json.dumps({'scope':'One150 run per config, original stochastic sampling; same E1 runtime for GC/Humming, different activation quantization/finalization/adapter paths remain','groups':report},indent=2));print({k:v['completion_tokens'] for k,v in report.items()})
