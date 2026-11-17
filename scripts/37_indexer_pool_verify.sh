#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=/root/dsk_opt_goal_20261116/src:/root/dsk_opt_goal_20261116/src/indexer
export TRITON_CACHE_DIR=/root/dsk_opt_goal_20261116/cache/indexer-triton
export PYTHONUNBUFFERED=1
exec /root/dsk_opt_goal_20261116/.venv-e0/bin/python -m torch.distributed.run --standalone --nproc-per-node=8 /root/dsk_opt_goal_20261116/scripts/36_indexer_pool_verify.py
