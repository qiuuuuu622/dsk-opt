#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_runtime_env.sh"
export DYN_SYSTEM_PORT=8081
export DSK_AUDIT_DIR=/root/dsk_opt_goal_20261116/groups/P-GC-ALL-w4a8-r2/evidence/loaded-layer-configs
exec /root/gc_moe_isolated_20261116/.venv/bin/python -m dynamo.vllm --worker-cls dsk_gc_worker.Worker \
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
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/root/dsk_opt_goal_20261116/groups/P-GC-ALL-w4a8-r2/traces","torch_profiler_with_stack":false,"torch_profiler_record_shapes":true,"ignore_frontend":true,"delay_iterations":5,"max_iterations":20}'
