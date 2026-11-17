"""Classify post-replay diagnostic kernels; never claim full-workload hit rate."""
import json,pathlib,argparse
p=argparse.ArgumentParser();p.add_argument('--group',required=True);a=p.parse_args()
r=pathlib.Path('/root/dsk_opt_goal_20261116');g=r/'groups'/a.group
d=json.loads((g/'evidence/profile-diagnostic.json').read_text())
checks={
 'humming_w4a8':lambda n:n.startswith('void humming<') and 'FloatingPointType<8u, 4u, 3u>, FloatingPointType<4u, 2u, 1u>' in n,
 'tiled_finalize':lambda n:n=='_grouped' or n=='_indexed',
 'compact_indexer_scores':lambda n:'_candidate_wgmma_kernel' in n,
 'compact_candidate_publish':lambda n:'publish_candidate_blocks' in n,
 'head_row_exchange':lambda n:'exchange_kernel' in n,
}
rows=[]
for p in d['profiles']:
 counts={label:sum(v['count'] for n,v in p['kernels'].items() if match(n)) for label,match in checks.items()}
 rows.append({'file':p['file'],'counts':counts,'all_paths_observed':all(counts.values())})
report={'scope':'post-timing 3-request diagnostic only; these counts are not whole150 hit rates','all_8_ranks_all_paths_observed':len(rows)==8 and all(x['all_paths_observed'] for x in rows),'ranks':rows}
(g/'evidence/combined-path-proof.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
