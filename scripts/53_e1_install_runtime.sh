#!/usr/bin/env bash
set -euo pipefail
source /root/gc_moe_isolated_20261116/scripts/00_env.sh
[[ "$(cat "${GC_BUILD_EXIT_FILE:-$GC_ENV_ROOT/evidence/vllm-build-jobs24-exit.txt}")" == 0 ]] || { echo "Build not successful; refusing runtime mutation"; exit 1; }
if pgrep -f '^bash /root/dsk_opt_goal_20261116/scripts/(50_e1_build_jobs24|56_e1_build_elfutils|58_e1_build_headers).sh$' >/dev/null; then echo "Build still active"; exit 1; fi
uv pip freeze --python "$VIRTUAL_ENV/bin/python" > "$GC_ENV_ROOT/evidence/before-runtime-install.lock.txt"
uv pip install --python "$VIRTUAL_ENV/bin/python" --no-deps -r "$GC_ENV_ROOT/evidence/runtime-frozen-public.txt"
uv pip install --python "$VIRTUAL_ENV/bin/python" --no-deps --index-url https://download.pytorch.org/whl/cu130 -r "$GC_ENV_ROOT/evidence/runtime-frozen-cuda.txt"
uv pip freeze --python "$VIRTUAL_ENV/bin/python" > "$GC_ENV_ROOT/evidence/after-runtime-install.lock.txt"
set +e
uv pip check --python "$VIRTUAL_ENV/bin/python" > "$GC_ENV_ROOT/evidence/runtime-pip-check.txt" 2>&1
code=$?
set -e
printf '%s\n' "$code" > "$GC_ENV_ROOT/evidence/runtime-pip-check-exit.txt"
# Generate source-distribution metadata without recompiling or fetching wheels.
export CUDA_HOME=/usr/local/cuda
export VLLM_TARGET_DEVICE=cuda
export VLLM_VERSION_OVERRIDE=0.32.0
export VLLM_REQUIRE_RUST_FRONTEND=0
export VLLM_USE_PRECOMPILED=0
export VLLM_USE_PRECOMPILED_RUST=0
(cd "$GC_ENV_ROOT/src/vllm-5df63a7" && "$VIRTUAL_ENV/bin/python" setup.py egg_info)
# Metadata inconsistencies are recorded separately; actual imports must pass.
export PYTHONPATH="$GC_ENV_ROOT/src/vllm-5df63a7"
"$VIRTUAL_ENV/bin/python" /root/dsk_opt_goal_20261116/scripts/54_e1_runtime_smoke.py
