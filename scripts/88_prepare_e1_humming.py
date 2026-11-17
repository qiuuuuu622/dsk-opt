import pathlib,json,shutil
r=pathlib.Path('/root/dsk_opt_goal_20261116');old=r/'groups/P-FAST-HUM-ALL-r1';g=r/'groups/P-E1-FAST-HUM-ALL-r1';assert not g.exists()
for d in ['scripts','logs','results','evidence','traces','discovery']:(g/d).mkdir(parents=True)
for p in (old/'scripts').iterdir():
 if p.is_file():(g/'scripts'/p.name).write_text(p.read_text().replace(old.name,g.name).replace('p-fast-hum-all-r1','p-e1-fast-hum-all-r1'))
p=g/'scripts/00_runtime_env.sh';p.write_text(p.read_text().replace('#!/usr/bin/env bash','#!/usr/bin/env bash\nsource /root/gc_moe_isolated_20261116/scripts/00_env.sh'))
p=g/'scripts/01_worker.sh';s=p.read_text().replace('/root/dsk_opt_goal_20261116/.venv-e0/bin/python','/root/gc_moe_isolated_20261116/.venv/bin/python').replace('export PYTHONPATH=/root/dsk_opt_goal_20261116/src:', 'export PYTHONPATH=/root/gc_moe_isolated_20261116/src/vllm-5df63a7:/root/dsk_opt_goal_20261116/src:');p.write_text(s)
# Frontend stays exactly E0, no E1 env sourced into it.
s=(r/'groups/P-FAST-GC-ALL-r2/scripts/02_frontend.sh').read_text().replace('P-FAST-GC-ALL-r2',g.name).replace('p-fast-gc-all-r2','p-e1-fast-hum-all-r1');(g/'scripts/02_frontend.sh').write_text(s)
(g/'evidence/environment-delta.json').write_text(json.dumps({'purpose':'Same E1 GPU environment comparison to P-FAST-GC-ALL-r2; environment-only comparison to P-FAST-HUM-ALL-r1','worker':'Torch2.11.0+cu130 Triton3.7.1 CUTLASS4.8.1 NCCL2.28.9 vLLM0.32.0 Humming0.1.17','frontend':'exact E0 frontend same as GC and Humming fastokens groups','MoE':'Humming target-only FP8 E4M3 per-token / FP4','finalize':'same tiled finalizer as E0 Humming; not full scale/shared fusion','all_other_parameters':'copied from P-FAST-HUM-ALL-r1'},indent=2))
s=(r/'scripts/10_humming_finalize_correctness.py').read_text().replace('results/humming-finalize','results/e1-humming-finalize')
(r/'scripts/89_e1_humming_finalize.py').write_text(s)
s=(r/'scripts/85_run_fastokens_group.py').read_text();start=s.index('assert a.group in ');end=s.index('\n',start);s=s[:start]+"assert a.group=='P-E1-FAST-HUM-ALL-r1'"+s[end:]
short=""" stop_old()
 with (g/'logs/finalize-e1-check.log').open('x') as check_log:
  command='source /root/gc_moe_isolated_20261116/scripts/00_env.sh; export OMP_NUM_THREADS=2 PYTHONPATH=/root/gc_moe_isolated_20261116/src/vllm-5df63a7:/root/dsk_opt_goal_20261116/src; python -m torch.distributed.run --standalone --nproc-per-node=8 /root/dsk_opt_goal_20261116/scripts/89_e1_humming_finalize.py'
  result=subprocess.run(['bash','-c',command],stdout=check_log,stderr=subprocess.STDOUT,timeout=900)
  (g/'evidence/finalize-e1-exit.json').write_text(json.dumps({'exit_code':result.returncode}))
  assert result.returncode==0,'E1 finalizer correctness failed'
"""
s=s.replace(' stop_old()\n',short)
(r/'scripts/90_run_e1_humming.py').write_text(s)
s=(r/'scripts/87_fastokens_sequence_retry.py').read_text().replace("old='P-GC-ALL-w4a8-r4'","old='P-FAST-HUM-ALL-r1'").replace("[('P-FAST-GC-ALL-r2','P-GC-ALL-w4a8-r4'),('P-FAST-HUM-ALL-r1','P-ALL-w4a8-stride-r2')]","[('P-E1-FAST-HUM-ALL-r1','P-FAST-GC-ALL-r2')]").replace('85_run_fastokens_group.py','90_run_e1_humming.py').replace('evidence/fastokens-sequence.json','evidence/e1-humming-sequence.json')
(r/'scripts/91_e1_humming_sequence.py').write_text(s)
