#!/usr/bin/env bash
# Serve TensorBoard for every run under runs/ (each trainer logs to runs/<run>/tb).
#
#   scripts/tensorboard.sh                 # http://127.0.0.1:6006 on the GPU box
#   PORT=6007 HOST=0.0.0.0 scripts/tensorboard.sh
#
# From your laptop, tunnel instead of exposing the port:
#   ssh -N -L 6006:127.0.0.1:6006 <gpu-host>   then open http://localhost:6006

set -euo pipefail
cd "$(dirname "$0")/.."
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-6006}
mkdir -p runs
echo "TensorBoard on http://$HOST:$PORT (ssh -N -L $PORT:127.0.0.1:$PORT <gpu-host>)"
exec uv run --group train tensorboard --logdir runs --host "$HOST" --port "$PORT" \
  --reload_interval 30
