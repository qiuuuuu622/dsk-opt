#!/usr/bin/env bash
set -euo pipefail
r=/root/dsk_opt_goal_20261116
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
export PYTHONPATH="$r/src/headrow"
export DSV41_HEAD_EXCHANGE_LIBRARY="$r/build/headrow/libsegment_exchange.so"
cd "$r/src/headrow"
"$r/.venv-e0/bin/python" -m vllm_dsv41_head_exchange.build --nvcc /usr/local/cuda/bin/nvcc
exec timeout --signal=TERM --kill-after=30s 1200s "$r/.venv-e0/bin/python" -m torch.distributed.run --standalone --nproc-per-node=8 --module benchmarks.kernels.head_exchange.verify --rows 1 7 8 9 32 64 127 128 129 256 512 --iterations 200 --output "$r/results/headrow-tp8-verify.json"
