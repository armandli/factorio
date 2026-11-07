#!/usr/bin/env bash
# Stop a background training run (the trainer saves a checkpoint first).
#
#   scripts/stop.sh ppo-20261106-120000
#   scripts/stop.sh --all

set -euo pipefail
cd "$(dirname "$0")/.."

stop() {
  local pid_file="runs/$1/train.pid" pid
  [[ -f "$pid_file" ]] || { echo "no pid file for run '$1'"; return 1; }
  pid=$(cat "$pid_file")
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM "$pid"
    echo "sent SIGTERM to run '$1' (pid $pid); it stops after saving a checkpoint"
  else
    echo "run '$1' is not running"
  fi
}

[[ $# -eq 1 ]] || { echo "usage: $0 <run-name> | --all" >&2; exit 2; }
if [[ "$1" == --all ]]; then
  for f in runs/*/train.pid; do
    [[ -e "$f" ]] && stop "$(basename "$(dirname "$f")")" || true
  done
else
  stop "$1"
fi
