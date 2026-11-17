#!/usr/bin/env bash
set -euo pipefail
export DYN_DISCOVERY_BACKEND=file DYN_REQUEST_PLANE=tcp DYN_EVENT_PLANE=zmq
export DYN_FILE_KV=/root/dsk_opt_goal_20261116/groups/P-GC-ALL-w4a8-r4/discovery
export DYN_NAMESPACE=dsk-gc-all-w4a8-r4 PYTHONUNBUFFERED=1
unset PYTHONPATH
exec /root/dsk_opt_goal_20261116/.venv-e0/bin/python -m dynamo.frontend --dyn-chat-processor vllm --http-host 127.0.0.1 --http-port 8000 --discovery-backend file --request-plane tcp --event-plane zmq
