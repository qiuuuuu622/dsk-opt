#!/usr/bin/env bash
set -euo pipefail
source /root/gc_moe_isolated_20261116/scripts/00_env.sh
export GC_MOE_PTXAS=/root/gc_moe_isolated_20261116/tools/cuda134/nvidia/cu13/bin/ptxas
export GIT_EXEC_PATH="$GC_ENV_ROOT/tools/git/usr/lib/git-core"
export PATH="$GC_ENV_ROOT/tools/git/usr/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES=""
export VLLM_TARGET_DEVICE=cuda
export TORCH_CUDA_ARCH_LIST=9.0a
export CPATH="$GC_ENV_ROOT/tools/elfutils/sysroot/usr/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$GC_ENV_ROOT/tools/elfutils/sysroot/usr/lib/x86_64-linux-gnu${LIBRARY_PATH:+:$LIBRARY_PATH}"
export MAX_JOBS=24
export NVCC_THREADS=1
export SETUPTOOLS_SCM_PRETEND_VERSION=0.32.0
export VLLM_REQUIRE_RUST_FRONTEND=0
export VLLM_USE_PRECOMPILED=0
export VLLM_USE_PRECOMPILED_RUST=0
export FETCHCONTENT_BASE_DIR="$GC_ENV_ROOT/cache/vllm-deps"
cd "$GC_ENV_ROOT/src/vllm-5df63a7"
set +e
"$VIRTUAL_ENV/bin/python" setup.py build_ext --inplace
code=$?
printf '%s\n' "$code" > "$GC_ENV_ROOT/evidence/vllm-build-elfutils-exit.txt"
exit "$code"
