#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_runtime_env.sh"
# Run only after checking GPU ownership, installed Dynamo, and frontend readiness.
export DYN_SYSTEM_PORT=8081
export PYTHONPATH=/root/gc_moe_isolated_20261116/src/vllm-5df63a7:/root/dsk_opt_goal_20261116/src:/root/dsk_opt_goal_20261116/src/indexer:/root/dsk_opt_goal_20261116/src/headrow
export DSK_AUDIT_DIR=/root/dsk_opt_goal_20261116/groups/P-E1-FAST-HUM-ALL-r1/evidence/loaded-layer-configs
/root/gc_moe_isolated_20261116/.venv/bin/python - <<'CHECK'
import pathlib,json
r=pathlib.Path('/root/dsk_opt_goal_20261116')
files=list((r/'results/headrow-adapter').glob('rank*.json'))
assert len(files)==8
assert json.loads((r/'evidence/indexer-strided-exit.json').read_text())['exit_code']==0
assert all(json.loads(p.read_text())['passed'] for p in files)
assert json.loads((r/'groups/QI-compact-r1/evidence/quality-gate.json').read_text())['status']=='structural_proxy_pass'
CHECK
exec /root/gc_moe_isolated_20261116/.venv/bin/python /root/dsk_opt_goal_20261116/src/dsk_fastokens_entry.py dynamo.vllm --worker-cls dsk_combined_worker.Worker \
  --model /model \
  --served-model-name sf-finance-dsk41-dynamo-vllm-alyeast-prod \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --engram-config '{"cpu_offload":true}' \
  --max-num-seqs 64 \
  --max-cudagraph-capture-size 64 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.92 \
  --block-size 64 \
  --mm-encoder-tp-mode data \
  --tokenizer-mode deepseek_v41 \
  --enable-multimodal \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":true}' \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/root/dsk_opt_goal_20261116/groups/P-E1-FAST-HUM-ALL-r1/traces","torch_profiler_with_stack":false,"torch_profiler_record_shapes":true,"ignore_frontend":true,"delay_iterations":5,"max_iterations":20}'
