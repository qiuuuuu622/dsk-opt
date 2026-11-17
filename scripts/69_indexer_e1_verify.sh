#!/usr/bin/env bash
set -uo pipefail
source /root/gc_moe_isolated_20261116/scripts/00_env.sh
export OMP_NUM_THREADS=2
export PYTHONPATH=/root/gc_moe_isolated_20261116/src/vllm-5df63a7:/root/dsk_opt_goal_20261116/src:/root/dsk_opt_goal_20261116/src/indexer
python -m torch.distributed.run --standalone --nproc-per-node=8 /root/dsk_opt_goal_20261116/scripts/68_indexer_strided_e1_verify.py
code=$?
printf '%s\n' "$code" > /root/dsk_opt_goal_20261116/evidence/indexer-strided-e1-exit.txt
exit "$code"
