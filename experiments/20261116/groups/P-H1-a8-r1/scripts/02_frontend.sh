#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/00_runtime_env.sh"
exec /root/dsk_opt_goal_20261116/.venv-e0/bin/python -m dynamo.frontend \
  --dyn-chat-processor vllm --http-host 127.0.0.1 --http-port 8000 \
  --discovery-backend file --request-plane tcp --event-plane zmq
