#!/usr/bin/env bash
set -euo pipefail
source /root/gc_moe_isolated_20261116/scripts/00_env.sh
export OMP_NUM_THREADS=2
export GC_MOE_PTXAS=/root/gc_moe_isolated_20261116/tools/cuda134/nvidia/cu13/bin/ptxas
mkdir -p "$XDG_CACHE_HOME/torch/kernels"
r=/root/dsk_opt_goal_20261116
cd /root/gc_moe_isolated_20261116/src/gc-moe
exec timeout --signal=TERM --kill-after=30s 1800s python -m torch.distributed.run --standalone --nproc-per-node=8 "$r/scripts/03_quant_utils_harness.py" tests/test_sm90_gc_moe_readiness.py --num-processes 8 --mode eager --steps 10 --num-tokens 64 --output-dir "$r/results/gc-eager-smoke-ptx134"
