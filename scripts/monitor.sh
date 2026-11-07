#!/usr/bin/env bash
# Training dashboard for a headless GPU box: GPU, Factorio containers, trainer processes,
# per-run progress (factorio-monitor) and the tail of each active run's log.
#
#   scripts/monitor.sh                  # refresh every 30 s (INTERVAL=10 for faster)
#   scripts/monitor.sh --once           # print once, e.g. from cron or over ssh
#   scripts/monitor.sh runs/ppo-...     # one run in detail (return curve, latest metrics)

set -uo pipefail
cd "$(dirname "$0")/.."

ONCE=0
TARGET=runs
for arg in "$@"; do
  case $arg in
    --once) ONCE=1 ;;
    *) TARGET=$arg ;;
  esac
done
INTERVAL=${INTERVAL:-30}
STALE_AFTER=${STALE_AFTER:-900}  # seconds without progress before a run is flagged

section() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

dashboard() {
  echo "$(hostname)  $(date '+%F %T')  load: $(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || uptime)"

  if command -v nvidia-smi >/dev/null; then
    section "GPU"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
      --format=csv,noheader
  fi

  section "Factorio containers"
  if docker info >/dev/null 2>&1; then
    docker stats --no-stream --format 'table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}' \
      $(docker ps -q --filter name=cluster-factorio_) 2>/dev/null \
      || echo "(none running)"
  else
    echo "docker daemon is not running"
  fi

  section "Trainer processes"
  pgrep -af "factorio-train" | grep -v pgrep || echo "(none)"

  section "Runs"
  uv run --group train factorio-monitor "$TARGET" --stale-after "$STALE_AFTER"

  # last log lines of runs that wrote metrics recently
  local log
  for log in $(find runs -maxdepth 2 -name train.log -mmin -30 2>/dev/null | sort); do
    section "$(dirname "$log" | xargs basename): tail"
    tail -n 3 "$log" | cut -c1-200
  done
}

if ((ONCE)); then
  dashboard
  exit 0
fi
while true; do
  out=$(dashboard 2>&1)
  clear
  echo "$out"
  echo
  echo "(refreshing every ${INTERVAL}s, Ctrl-C to quit)"
  sleep "$INTERVAL"
done
