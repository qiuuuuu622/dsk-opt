#!/usr/bin/env bash
set -uo pipefail
source /root/gc_moe_isolated_20261116/scripts/00_env.sh
export OMP_NUM_THREADS=2
export PYTHONPATH=/root/gc_moe_isolated_20261116/src/vllm-5df63a7:/root/dsk_opt_goal_20261116/src:/root/dsk_opt_goal_20261116/src/indexer
python -c 'import triton,torch; from triton.experimental.gluon.language.nvidia.hopper import mbarrier; assert triton.__version__=="3.7.1"; assert hasattr(mbarrier,"allocate_mbarrier"); print(torch.__version__,triton.__version__,flush=True)' || exit 1
python -m torch.distributed.run --standalone --nproc-per-node=8 /root/dsk_opt_goal_20261116/scripts/70_gc_adapter_triton37_verify.py
code=$?
printf '%s\n' "$code" > /root/dsk_opt_goal_20261116/evidence/gc-service-adapter-triton37-exit.txt
[[ "$code" == 0 ]] || exit "$code"
python -m torch.distributed.run --standalone --nproc-per-node=8 /root/dsk_opt_goal_20261116/scripts/68_indexer_strided_e1_verify.py
code=$?
printf '%s\n' "$code" > /root/dsk_opt_goal_20261116/evidence/indexer-strided-e1-exit.txt
exit "$code"
