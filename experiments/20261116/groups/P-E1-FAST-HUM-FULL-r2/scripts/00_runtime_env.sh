#!/usr/bin/env bash
source /root/gc_moe_isolated_20261116/scripts/00_env.sh
# Isolated single-host discovery; no existing etcd/NATS service is needed.
export DYN_DISCOVERY_BACKEND=file
export DYN_REQUEST_PLANE=tcp
export DYN_EVENT_PLANE=zmq
export DYN_FILE_KV=/root/dsk_opt_goal_20261116/groups/P-E1-FAST-HUM-FULL-r2/discovery
export DYN_NAMESPACE=p-e1-fast-hum-full-r2
export PYTHONUNBUFFERED=1

export DSK_HUMMING_FINALIZE=1
export DSK_HUMMING_A8=1

export DSK_COMPACT_INDEXER=1
export DSK_HEADROW=1
export DSK_INDEXER_EVIDENCE_DIR=/root/dsk_opt_goal_20261116/groups/P-E1-FAST-HUM-FULL-r2/evidence/compact
export DSK_HEADROW_EVIDENCE_DIR=/root/dsk_opt_goal_20261116/groups/P-E1-FAST-HUM-FULL-r2/evidence/headrow
export DSV41_HEAD_EXCHANGE_LIBRARY=/root/dsk_opt_goal_20261116/build/headrow/libsegment_exchange.so

export VLLM_USE_FASTOKENS=1
export DYN_TOKENIZER=fastokens DYN_TOKENIZER_FALLBACK=false
export DSK_FASTOKENS_EVIDENCE_DIR=/root/dsk_opt_goal_20261116/groups/P-E1-FAST-HUM-FULL-r2/evidence/fastokens

export DSK_HUMMING_FULL_FINALIZE=1
export DSK_FULL_FINALIZE_EVIDENCE_DIR=/root/dsk_opt_goal_20261116/groups/P-E1-FAST-HUM-FULL-r2/evidence/full-finalize
