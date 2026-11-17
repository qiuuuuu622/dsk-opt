#!/usr/bin/env bash
set -euo pipefail
r=/root/dsk_opt_goal_20261116
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2
export TRITON_CACHE_DIR="$r/cache/e0-triton"
exec timeout --signal=TERM --kill-after=30s 900s "$r/.venv-e0/bin/python" -m torch.distributed.run --standalone --nproc-per-node=8 "$r/scripts/10_humming_finalize_correctness.py"
