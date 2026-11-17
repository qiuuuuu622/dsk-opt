#!/usr/bin/env bash
set -euo pipefail
source /root/gc_moe_isolated_20261116/scripts/00_env.sh
export OMP_NUM_THREADS=2
export GC_MOE_PTXAS=/root/gc_moe_isolated_20261116/tools/cuda134/nvidia/cu13/bin/ptxas
mkdir -p "$XDG_CACHE_HOME/torch/kernels"
cd /root/gc_moe_isolated_20261116/src/gc-moe
exec python /root/dsk_opt_goal_20261116/scripts/07_gc_suite.py
