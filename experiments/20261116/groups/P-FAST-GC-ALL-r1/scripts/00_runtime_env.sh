#!/usr/bin/env bash
source /root/gc_moe_isolated_20261116/scripts/00_env.sh
export PYTHONPATH=/root/gc_moe_isolated_20261116/src/vllm-5df63a7:/root/dsk_opt_goal_20261116/src:/root/dsk_opt_goal_20261116/src/indexer:/root/dsk_opt_goal_20261116/src/headrow
export DYN_DISCOVERY_BACKEND=file
export DYN_REQUEST_PLANE=tcp
export DYN_EVENT_PLANE=zmq
export DYN_FILE_KV=/root/dsk_opt_goal_20261116/groups/P-FAST-GC-ALL-r1/discovery
export DYN_NAMESPACE=p-fast-gc-all-r1
export PYTHONUNBUFFERED=1
export DSK_HUMMING_FINALIZE=0
export DSK_HUMMING_A8=0
export DSK_COMPACT_INDEXER=1
export DSK_HEADROW=1
export DSK_GC_MOE=1
export DSK_GC_FINALIZE=1
export DSK_GC_EVIDENCE_DIR=/root/dsk_opt_goal_20261116/groups/P-FAST-GC-ALL-r1/evidence/gc
export DSK_INDEXER_EVIDENCE_DIR=/root/dsk_opt_goal_20261116/groups/P-FAST-GC-ALL-r1/evidence/compact
export DSK_HEADROW_EVIDENCE_DIR=/root/dsk_opt_goal_20261116/groups/P-FAST-GC-ALL-r1/evidence/headrow
export DSV41_HEAD_EXCHANGE_LIBRARY=/root/dsk_opt_goal_20261116/build/headrow/libsegment_exchange.so

export VLLM_USE_FASTOKENS=1
export DYN_TOKENIZER=fastokens DYN_TOKENIZER_FALLBACK=false
export DSK_FASTOKENS_EVIDENCE_DIR=/root/dsk_opt_goal_20261116/groups/P-FAST-GC-ALL-r1/evidence/fastokens
