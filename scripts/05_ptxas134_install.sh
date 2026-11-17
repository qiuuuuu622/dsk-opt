#!/usr/bin/env bash
set -euo pipefail
source /root/gc_moe_isolated_20261116/scripts/00_env.sh
uv pip install --python "$VIRTUAL_ENV/bin/python" --target /root/gc_moe_isolated_20261116/tools/cuda134 --no-deps --index-url https://pypi.org/simple nvidia-cuda-nvcc==13.4.92
