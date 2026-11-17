#!/usr/bin/env bash
set -euo pipefail
r=/root/dsk_opt_goal_20261116
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
export PYTHONPATH="$r/src/indexer"
export TRITON_CACHE_DIR="$r/cache/indexer-triton"
cd "$r/src/indexer"
exec timeout --signal=TERM --kill-after=30s 1200s "$r/.venv-e0/bin/python" -m torch.distributed.run --standalone --nproc-per-node=8 --module benchmarks.kernels.dsv41_indexer.verify --distributed --output "$r/results/indexer-tp8-verify.json"
