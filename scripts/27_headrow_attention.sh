#!/usr/bin/env bash
set -euo pipefail
r=/root/dsk_opt_goal_20261116
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
export PYTHONPATH="$r/src/headrow"
export DSV41_HEAD_EXCHANGE_LIBRARY="$r/build/headrow/libsegment_exchange.so"
cd "$r/src/headrow"
"$r/.venv-e0/bin/python" - <<'PYCODE'
from pathlib import Path
from importlib.metadata import distribution
p=Path(distribution('vllm').locate_file('vllm/third_party/flashmla/flash_mla_interface.py'))
out=Path('/root/dsk_opt_goal_20261116/evidence/flash_mla_interface.py')
out.write_text(p.read_text().replace('import vllm._flashmla_C\n',''))
PYCODE
exec timeout --signal=TERM --kill-after=30s 1800s "$r/.venv-e0/bin/python" -m torch.distributed.run --standalone --nproc-per-node=8 --module benchmarks.kernels.head_exchange.benchmark --transport fused --rows 8 64 128 512 --ratios 0 1 2 --context 32768 --vary-lengths --repeat 1 --inner 1 --interface "$r/evidence/flash_mla_interface.py" --output "$r/results/headrow-attention-pilot.json"
