# Shared helpers for the train_*.sh scripts. Source this file, don't run it.
#
# Common environment variables (each train_*.sh sets its own defaults):
#   RUN_NAME          run directory under runs/ (reuse one to resume it)  [<algo>-<timestamp>]
#   NUM_ENVS          parallel envs, one Factorio container each
#   FIRST_CONTAINER   env i uses container FIRST_CONTAINER+i (separate runs need disjoint ranges)
#   TOTAL_STEPS       env steps summed over all envs
#   ENV_ID            FLE task id                                          [iron_ore_throughput]
#   MAX_EPISODE_STEPS steps per episode                                    [50]
#   ERROR_PENALTY     reward subtracted when a program errors              [0.1]
#   DEVICE            torch device                                         [cuda]
#   MAX_RESTARTS      restart-and-resume attempts after a crash            [20]
#   DETACH=1          run in the background (logs to runs/<RUN_NAME>/train.log)

set -euo pipefail

# the calling train_*.sh and its own arguments, for the DETACH re-exec (before the cd below)
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
SCRIPT_ARGS=("$@")
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log() { echo "[$(date '+%F %T')] $*"; }
die() { log "error: $*" >&2; exit 1; }

# Compose names containers <project>-factorio_<i>-1, where <project> is the directory holding
# FLE's compose file ("cluster" for fle/cluster/ via `fle cluster start`; "fle" if it lives in
# ~/.local/state/fle), so match any project prefix.
FACTORIO_NAME_RE='^[A-Za-z0-9_.-]+-factorio_[0-9]+-1$'
container_name() {
  docker ps -a --format '{{.Names}}' | grep -E "^[A-Za-z0-9_.-]+-factorio_$1-1\$" | head -1 || true
}
container_running() {
  docker ps --format '{{.Names}}' | grep -qE "^[A-Za-z0-9_.-]+-factorio_$1-1\$"
}

setup_python() {
  command -v uv >/dev/null || die "uv is not installed (https://docs.astral.sh/uv/)"
  uv sync --group train --quiet
}

check_gpu() {
  if [[ "${DEVICE:-cuda}" != cuda* ]]; then
    return
  fi
  if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader \
      | sed 's/^/  gpu /'
  fi
  uv run --group train python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null \
    || die "torch can't see a CUDA GPU; set DEVICE=cpu to train without one"
}

# Start containers 0..FIRST_CONTAINER+NUM_ENVS-1 if any of this run's are missing, then wait
# for their RCON ports. `fle cluster start` leaves running containers alone.
ensure_cluster() {
  local first=$1 n=$2 needed=$(($1 + $2)) missing=0 running
  command -v docker >/dev/null || die "docker is not installed"
  docker info >/dev/null 2>&1 || die "docker daemon is not running"
  for ((i = first; i < needed; i++)); do
    container_running "$i" || missing=1
  done
  if ((missing)); then
    running=$(docker ps --format '{{.Names}}' | grep -cE "$FACTORIO_NAME_RE" || true)
    local total=$((needed > running ? needed : running))
    log "starting Factorio cluster with $total container(s)"
    uv run fle cluster start -n "$total"
  fi
  wait_for_rcon "$first" "$n"
}

# wait_for_rcon <first> <n> [since]: only log lines after `since` count (docker logs keeps
# lines from before a restart).
wait_for_rcon() {
  local first=$1 n=$2 since=${3:-} deadline=$((SECONDS + 300)) name
  for ((i = first; i < first + n; i++)); do
    name=$(container_name "$i")
    [[ -n "$name" ]] || die "no container for Factorio instance $i"
    until docker logs ${since:+--since "$since"} "$name" 2>&1 \
        | grep -q "Starting RCON interface"; do
      ((SECONDS < deadline)) || die "$name did not open RCON within 5 minutes"
      sleep 3
    done
  done
  log "containers $first..$((first + n - 1)) ready"
}

restart_containers() {
  local first=$1 n=$2 names=() since
  for ((i = first; i < first + n; i++)); do names+=("$(container_name "$i")"); done
  log "restarting ${names[*]}"
  since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  docker restart "${names[@]}" >/dev/null
  wait_for_rcon "$first" "$n" "$since"
}

latest_periodic_checkpoint() {
  ls "$1"/checkpoint-*.pt 2>/dev/null | grep -E 'checkpoint-[0-9]+\.pt$' \
    | sort -t- -k2 -n | tail -1 || true
}

# train <algo> [factorio-train flags...]
# Runs factorio-train, and after a crash restarts this run's containers and resumes from the
# newest checkpoint, up to MAX_RESTARTS times. SIGINT/SIGTERM stop it cleanly (the trainer
# saves a checkpoint first).
train() {
  local algo=$1; shift
  RUN_NAME=${RUN_NAME:-$algo-$(date +%Y%m%d-%H%M%S)}
  local run_dir="runs/$RUN_NAME"
  mkdir -p "$run_dir"

  if [[ "${DETACH:-0}" == 1 ]]; then
    DETACH=0 RUN_NAME=$RUN_NAME nohup "$SCRIPT_PATH" ${SCRIPT_ARGS[@]+"${SCRIPT_ARGS[@]}"} >>"$run_dir/train.log" 2>&1 &
    echo $! >"$run_dir/train.pid"
    log "started $algo run '$RUN_NAME' in the background (pid $!)"
    log "  log:     tail -f $run_dir/train.log"
    log "  monitor: scripts/monitor.sh"
    log "  stop:    scripts/stop.sh $RUN_NAME"
    exit 0
  fi
  echo $$ >"$run_dir/train.pid"

  local first=${FIRST_CONTAINER:-0} n=${NUM_ENVS:-1} max_restarts=${MAX_RESTARTS:-20}
  local attempt=0 status child stopping=0
  trap 'stopping=1; [[ -n "${child:-}" ]] && kill -TERM "$child" 2>/dev/null' INT TERM

  log "run '$RUN_NAME': $algo on ${ENV_ID:-iron_ore_throughput}, envs $first..$((first + n - 1))"
  setup_python
  check_gpu
  ensure_cluster "$first" "$n"

  while true; do
    if [[ -f "$run_dir/checkpoint-final.pt" ]]; then
      log "run '$RUN_NAME' finished: $run_dir/checkpoint-final.pt"
      return 0
    fi
    local resume=() ckpt
    ckpt=$(latest_periodic_checkpoint "$run_dir")
    if [[ -n "$ckpt" ]]; then
      resume=(--resume "$ckpt")
      log "resuming from $ckpt"
    fi

    uv run --group train factorio-train "$algo" \
      --run-name "$RUN_NAME" \
      --num-envs "$n" \
      --first-container "$first" \
      --env-id "${ENV_ID:-iron_ore_throughput}" \
      --max-episode-steps "${MAX_EPISODE_STEPS:-50}" \
      --error-penalty "${ERROR_PENALTY:-0.1}" \
      --device "${DEVICE:-cuda}" \
      "$@" ${resume[@]+"${resume[@]}"} &
    child=$!
    status=0
    wait "$child" || status=$?
    # `wait` returns early when a trapped signal arrives; wait again for the trainer to save
    while kill -0 "$child" 2>/dev/null; do wait "$child" || status=$?; done
    child=

    if ((status == 0)); then
      continue  # checkpoint-final.pt now exists
    fi
    if ((stopping)); then
      log "stopped by signal; resume with: RUN_NAME=$RUN_NAME $SCRIPT_PATH"
      return 130
    fi
    attempt=$((attempt + 1))
    if ((attempt > max_restarts)); then
      die "factorio-train failed $attempt times (exit $status); giving up"
    fi
    log "factorio-train exited with $status (restart $attempt/$max_restarts in 30s)"
    sleep 30
    restart_containers "$first" "$n"
  done
}
