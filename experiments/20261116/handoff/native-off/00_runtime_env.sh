#!/usr/bin/env bash
# Isolated single-host discovery; no existing etcd/NATS service is needed.
export DYN_DISCOVERY_BACKEND=file
export DYN_REQUEST_PLANE=tcp
export DYN_EVENT_PLANE=zmq
export DYN_FILE_KV=/root/dsk_opt_goal_20261116/handoff/native-off/discovery
export DYN_NAMESPACE=dsk-restore-native-final
export PYTHONUNBUFFERED=1

export DSK_HUMMING_FINALIZE=0
export DSK_HUMMING_A8=0

export DSK_HUMMING_A8=0 DSK_HUMMING_FINALIZE=0 DSK_HUMMING_FULL_FINALIZE=0
export DSK_COMPACT_INDEXER=0 DSK_HEADROW=0 DSK_GC_MOE=0 DSK_GC_FINALIZE=0
export VLLM_USE_FASTOKENS=0 DYN_TOKENIZER=default DYN_TOKENIZER_FALLBACK=true
