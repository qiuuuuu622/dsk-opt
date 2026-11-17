#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_runtime_env.sh"
# Run only after checking GPU ownership, installed Dynamo, and frontend readiness.
export DSK_SAMPLING_AUDIT=/root/dsk_opt_goal_20261116/groups/Q0-native-r2/evidence/effective-sampling.jsonl
export PYTHONPATH=/root/dsk_opt_goal_20261116/src
export DSK_AUDIT_DIR=/root/dsk_opt_goal_20261116/groups/Q0-native-r2/evidence/loaded-layer-configs
exec /root/dsk_opt_goal_20261116/.venv-e0/bin/python -m dsk_sampling_audit_launch --worker-cls dsk_humming_worker.Worker \
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
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":true}'
