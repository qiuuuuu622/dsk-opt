#!/usr/bin/env bash
# Isolated single-host discovery; no existing etcd/NATS service is needed.
export DYN_DISCOVERY_BACKEND=file
export DYN_REQUEST_PLANE=tcp
export DYN_EVENT_PLANE=zmq
export DYN_FILE_KV=/root/bench_dsk_first150_20261116/discovery
export DYN_NAMESPACE=dsk-first150-20261116
export PYTHONUNBUFFERED=1
