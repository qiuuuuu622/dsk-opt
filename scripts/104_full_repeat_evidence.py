import pathlib,json,re
r=pathlib.Path('/root/dsk_opt_goal_20261116/groups');g=r/'P-E1-FAST-HUM-FULL-r2';base=r/'P-E1-FAST-HUM-ALL-r1'
def byrank(path):
 d=json.loads(path.read_text());return {int(re.search(r'_rank(\d+)\.',x['file']).group(1)):x for x in d['profiles']}
a=byrank(base/'evidence/profile-diagnostic.json');b=byrank(g/'evidence/profile-diagnostic.json');rows=[]
for rank in range(8):
 before=a[rank]['kernels'];after=b[rank]['kernels'];add=lambda k:sum(v['count'] for n,v in k.items() if 'CUDAFunctor_add<c10::BFloat16>' in n)
 counts={'humming_W4A8':sum(v['count'] for n,v in after.items() if n.startswith('void humming') and 'FloatingPointType<8u, 4u, 3u>, FloatingPointType<4u, 2u, 1u>' in n),'grouped_finalize':after['_grouped']['count'],'compact':sum(v['count'] for n,v in after.items() if '_candidate_wgmma_kernel' in n),'headrow':sum(v['count'] for n,v in after.items() if 'exchange_kernel' in n)}
 assert all(v>0 for v in counts.values());assert add(before)-add(after)==800
 hits=json.loads((g/f'evidence/full-finalize/branch-rank{rank}.json').read_text());assert len(hits)==40 and all(x['shared_fused'] and x['scale']==1. for x in hits.values())
 rows.append({'rank':rank,'counts':counts,'BF16_add_before':add(before),'BF16_add_after':add(after),'removed_shared_adds':800,'target_layers_observed':40,'actual_scale':1.0})
report={'scope':'Post-timing3request/20step short diagnostic; not whole150 hit rate. Cross-run kernel counts plus runtime branch audit; native draft unmodified.','all_four_paths':True,'shared_add_elimination_verified_all8':True,'ranks':rows}
(g/'evidence/full-fusion-proof.json').write_text(json.dumps(report,indent=2));(g/'evidence/four-paths.json').write_text(json.dumps(report,indent=2))
print('8 ranks verified,800 shared adds eliminated per rank')
