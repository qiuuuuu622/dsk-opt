#!/usr/bin/env bash
set -euo pipefail
source /root/gc_moe_isolated_20261116/scripts/00_env.sh
export GC_MOE_PTXAS=/root/gc_moe_isolated_20261116/tools/cuda134/nvidia/cu13/bin/ptxas
uv pip freeze --python "$VIRTUAL_ENV/bin/python" > "$GC_ENV_ROOT/evidence/before-vllm-build.lock.txt"
uv pip install --python "$VIRTUAL_ENV/bin/python" --constraint "$GC_ENV_ROOT/constraints.txt" 'cmake>=3.26.1' ninja 'packaging>=24.2' 'setuptools>=78.1.1,<81' 'setuptools-scm>=8' 'setuptools-rust>=1.9' wheel 'jinja2>=3.1.6' regex build
uv pip check --python "$VIRTUAL_ENV/bin/python"
uv pip freeze --python "$VIRTUAL_ENV/bin/python" > "$GC_ENV_ROOT/evidence/build-dependencies.lock.txt"
