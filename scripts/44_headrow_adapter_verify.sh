#!/usr/bin/env bash
set -euo pipefail
r=/root/dsk_opt_goal_20261116
export PYTHONPATH="$r/src:$r/src/headrow"
export DSV41_HEAD_EXCHANGE_LIBRARY="$r/build/headrow/libsegment_exchange.so"
export OMP_NUM_THREADS=1 PYTHONNOUSERSITE=1
unset PYTHONHOME
"$r/.venv-e0/bin/python" -m torch.distributed.run --standalone --nproc-per-node=8 "$r/scripts/43_headrow_adapter_verify.py"
