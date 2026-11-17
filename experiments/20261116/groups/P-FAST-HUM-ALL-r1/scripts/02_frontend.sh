#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_runtime_env.sh"
export VLLM_USE_FASTOKENS=1
export DYN_TOKENIZER=fastokens DYN_TOKENIZER_FALLBACK=false
export DSK_FASTOKENS_EVIDENCE_DIR=/root/dsk_opt_goal_20261116/groups/P-FAST-HUM-ALL-r1/evidence/fastokens
exec /root/dsk_opt_goal_20261116/.venv-e0/bin/python /root/dsk_opt_goal_20261116/src/dsk_fastokens_entry.py dynamo.frontend \
  --dyn-chat-processor vllm --http-host 127.0.0.1 --http-port 8000 \
  --discovery-backend file --request-plane tcp --event-plane zmq
