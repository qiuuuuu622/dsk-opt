#!/usr/bin/env bash
set -euo pipefail
export DYN_DISCOVERY_BACKEND=file DYN_REQUEST_PLANE=tcp DYN_EVENT_PLANE=zmq
export DYN_FILE_KV=/root/dsk_opt_goal_20261116/groups/P-FAST-GC-ALL-r1/discovery
export DYN_NAMESPACE=p-fast-gc-all-r1 PYTHONUNBUFFERED=1
unset PYTHONPATH
export VLLM_USE_FASTOKENS=1
export DYN_TOKENIZER=fastokens DYN_TOKENIZER_FALLBACK=false
export DSK_FASTOKENS_EVIDENCE_DIR=/root/dsk_opt_goal_20261116/groups/P-FAST-GC-ALL-r1/evidence/fastokens
exec /root/dsk_opt_goal_20261116/.venv-e0/bin/python /root/dsk_opt_goal_20261116/src/dsk_fastokens_entry.py dynamo.frontend --dyn-chat-processor vllm --http-host 127.0.0.1 --http-port 8000 --discovery-backend file --request-plane tcp --event-plane zmq
