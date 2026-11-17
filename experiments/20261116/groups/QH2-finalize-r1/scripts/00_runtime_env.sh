#!/usr/bin/env bash
# Isolated single-host discovery; no existing etcd/NATS service is needed.
export DYN_DISCOVERY_BACKEND=file
export DYN_REQUEST_PLANE=tcp
export DYN_EVENT_PLANE=zmq
export DYN_FILE_KV=/root/dsk_opt_goal_20261116/groups/QH2-finalize-r1/discovery
export DYN_NAMESPACE=dsk-qh2-finalize-r1
export PYTHONUNBUFFERED=1

export DSK_HUMMING_FINALIZE=1
export DSK_HUMMING_A8=0
