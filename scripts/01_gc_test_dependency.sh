#!/usr/bin/env bash
set -euo pipefail
source /root/gc_moe_isolated_20261116/scripts/00_env.sh
uv pip install --python "$VIRTUAL_ENV/bin/python" --no-deps --index-url https://pypi.org/simple 'deep-gemm==2.8.2'
uv pip check --python "$VIRTUAL_ENV/bin/python"
python -c 'import deep_gemm, deep_gemm.utils; print(deep_gemm.__file__); print(deep_gemm.utils.per_token_cast_to_fp4, deep_gemm.utils.per_token_cast_to_fp8)'
