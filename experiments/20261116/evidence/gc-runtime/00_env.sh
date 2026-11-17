#!/usr/bin/env bash
export GC_ENV_ROOT=/root/gc_moe_isolated_20261116
unset PYTHONPATH PYTHONHOME UV_OVERRIDE UV_CONSTRAINT
export VIRTUAL_ENV="$GC_ENV_ROOT/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export UV_CACHE_DIR="$GC_ENV_ROOT/cache/uv"
export TRITON_CACHE_DIR="$GC_ENV_ROOT/cache/triton"
export TORCH_EXTENSIONS_DIR="$GC_ENV_ROOT/cache/torch_extensions"
export XDG_CACHE_HOME="$GC_ENV_ROOT/cache/xdg"
export GC_MOE_PTXAS="$GC_ENV_ROOT/tools/cuda134/nvidia/cu13/bin/ptxas"
