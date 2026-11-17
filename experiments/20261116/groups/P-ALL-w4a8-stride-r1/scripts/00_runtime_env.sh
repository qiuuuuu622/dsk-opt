#!/usr/bin/env bash
# Isolated single-host discovery; no existing etcd/NATS service is needed.
export DYN_DISCOVERY_BACKEND=file
export DYN_REQUEST_PLANE=tcp
export DYN_EVENT_PLANE=zmq
export DYN_FILE_KV=/root/dsk_opt_goal_20261116/groups/P-ALL-w4a8-stride-r1/discovery
export DYN_NAMESPACE=dsk-p-all-w4a8-stride-r1
export PYTHONUNBUFFERED=1

export DSK_HUMMING_FINALIZE=1
export DSK_HUMMING_A8=1

export DSK_COMPACT_INDEXER=1
export DSK_HEADROW=1
export DSK_INDEXER_EVIDENCE_DIR=/root/dsk_opt_goal_20261116/groups/P-ALL-w4a8-stride-r1/evidence/compact
export DSK_HEADROW_EVIDENCE_DIR=/root/dsk_opt_goal_20261116/groups/P-ALL-w4a8-stride-r1/evidence/headrow
export DSV41_HEAD_EXCHANGE_LIBRARY=/root/dsk_opt_goal_20261116/build/headrow/libsegment_exchange.so
