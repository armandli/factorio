#!/usr/bin/env bash
# Train several algorithms side by side, pick the most sample-efficient one, and give it every
# container to train on. Runs unattended; re-running it with the same STAMP picks up where it
# left off (it skips runs that are already going and resumes ones that died).
#
#   phase 1  each algorithm in ALGOS trains on NUM_ENVS containers (+1 eval) for PHASE1_HOURS,
#            and at least until each has PHASE1_MIN_EVALS evaluations (hard cap 1.5x hours)
#   pick     `factorio-compare --metric $SELECT_METRIC` ranks the runs by return per env step
#            over the common step budget (see rl/compare.py) and writes the winner to winner.json
#   phase 2  the other runs are stopped, and the winner resumes on FINAL_NUM_ENVS containers
#            for PHASE2_HOURS (it keeps training afterwards unless STOP_AFTER=1)
#
#   DETACH=1 scripts/tournament.sh
#   STAMP=20261111-075642 DETACH=1 scripts/tournament.sh     # adopt / resume a tournament
#   ARGS_dqn="--learning-starts 400" PHASE1_HOURS=3 scripts/tournament.sh
#
# Settings (environment variables; train_*.sh settings such as EVAL_EVERY pass through):
#   ALGOS              algorithms to compare                         [ppo dqn mcgs]
#   NUM_ENVS           training envs per algorithm in phase 1        [4]
#   FINAL_NUM_ENVS     training envs for the winner in phase 2       [all phase-1 containers - 1]
#   PHASE1_HOURS       comparison time                               [2.5]
#   PHASE1_MIN_EVALS   evaluations each run needs before picking     [4]
#   PHASE2_HOURS       winner training time                          [6]
#   SELECT_METRIC      train | eval | auto, see factorio-compare     [train]
#                      (train: greedy DQN/MCGS evals can loop on one action, so eval_auc
#                      compares eval modes more than learning; see rl/compare.py)
#   CHECK_EVERY        seconds between monitor snapshots / health checks [600]
#   ARGS_<algo>        extra factorio-train flags for that algorithm
# Logs: runs/tournament-<STAMP>/{tournament.log,monitor.log,compare.txt,winner.json}

source "$(dirname "$0")/lib.sh"

STAMP=${STAMP:-$(date +%Y%m%d-%H%M%S)}
T_DIR="runs/tournament-$STAMP"
mkdir -p "$T_DIR"
if [[ "${DETACH:-0}" == 1 ]]; then
  DETACH=0 STAMP=$STAMP nohup "$SCRIPT_PATH" ${SCRIPT_ARGS[@]+"${SCRIPT_ARGS[@]}"} \
    >>"$T_DIR/tournament.log" 2>&1 &
  echo $! >"$T_DIR/tournament.pid"
  log "tournament $STAMP started in the background (pid $!)"
  log "  log: tail -f $T_DIR/tournament.log    compare: cat $T_DIR/compare.txt"
  exit 0
fi
echo $$ >"$T_DIR/tournament.pid"

read -ra ALGOS <<<"${ALGOS:-ppo dqn mcgs}"
export NUM_ENVS=${NUM_ENVS:-4} EVAL_EVERY=${EVAL_EVERY:-400} EVAL_EPISODES=${EVAL_EPISODES:-2}
export TOTAL_STEPS=${TOTAL_STEPS:-1000000}
PER_RUN=$(run_containers)
FINAL_NUM_ENVS=${FINAL_NUM_ENVS:-$((${#ALGOS[@]} * PER_RUN - 1))}
PHASE1_HOURS=${PHASE1_HOURS:-2.5} PHASE2_HOURS=${PHASE2_HOURS:-6}
PHASE1_MIN_EVALS=${PHASE1_MIN_EVALS:-4} CHECK_EVERY=${CHECK_EVERY:-600}
SELECT_METRIC=${SELECT_METRIC:-train}

state_get() { [[ -f "$T_DIR/state.$1" ]] && cat "$T_DIR/state.$1" || true; }
state_set() { echo "$2" >"$T_DIR/state.$1"; }
hours_since() { awk -v s="$1" -v n="$(date +%s)" 'BEGIN { printf "%.2f", (n - s) / 3600 }'; }
ge() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a >= b) }'; }

run_alive() {
  local pid_file="runs/$1/train.pid"
  [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null
}

# launch <algo> <run-name> <first-container> <num-envs>: start (or resume) a run unless it's going
launch() {
  local algo=$1 name=$2 first=$3 n=$4 extra_var="ARGS_$1"
  if run_alive "$name"; then
    return
  fi
  log "launching $name on containers $first..$((first + n)) (envs $n)"
  # shellcheck disable=SC2086  # ARGS_<algo> is a list of flags
  FIRST_CONTAINER=$first NUM_ENVS=$n RUN_NAME=$name DETACH=1 \
    "$REPO_ROOT/scripts/train_$algo.sh" ${!extra_var:-} >/dev/null
}

stop_and_wait() {
  local name=$1 pid
  run_alive "$name" || return 0
  pid=$(cat "runs/$name/train.pid")
  log "stopping $name"
  "$REPO_ROOT/scripts/stop.sh" "$name" >/dev/null || true
  for _ in $(seq 120); do kill -0 "$pid" 2>/dev/null || return 0; sleep 5; done
  log "warning: $name did not stop within 10 minutes; killing it"
  pkill -KILL -P "$pid" 2>/dev/null || true
  kill -KILL "$pid" 2>/dev/null || true
}

snapshot() {
  {
    echo "################ $(date '+%F %T') phase $(state_get phase)"
    "$REPO_ROOT/scripts/monitor.sh" --once 2>&1 | grep -v '^\s*$'
  } >>"$T_DIR/monitor.log"
}

compare_runs() {
  local runs=()
  for algo in "${ALGOS[@]}"; do runs+=("runs/$algo-$STAMP"); done
  uv run --group train factorio-compare --metric "$SELECT_METRIC" "${runs[@]}" "$@"
}

min_evals() {
  # factorio-compare exits 1 until there is a winner; only its JSON matters here
  { compare_runs --json 2>/dev/null || true; } | uv run python -c '
import json, sys
try:
    runs = json.load(sys.stdin)["runs"]
except Exception:
    runs = []
print(min((d["n_evals"] for d in runs), default=0) if len(runs) == '"${#ALGOS[@]}"' else 0)'
}

setup_python
[[ -n "$(state_get phase)" ]] || { state_set phase 1; state_set phase1_start "$(date +%s)"; }

if [[ "$(state_get phase)" == 1 ]]; then
  ensure_cluster 0 $((${#ALGOS[@]} * PER_RUN))
  log "phase 1: ${ALGOS[*]} on $NUM_ENVS envs each for ${PHASE1_HOURS}h (min $PHASE1_MIN_EVALS evals)"
  while true; do
    for i in "${!ALGOS[@]}"; do
      launch "${ALGOS[$i]}" "${ALGOS[$i]}-$STAMP" $((i * PER_RUN)) "$NUM_ENVS"
    done
    snapshot
    compare_runs >"$T_DIR/compare.txt" 2>&1 || true
    elapsed=$(hours_since "$(state_get phase1_start)")
    evals=$(min_evals || echo 0)
    log "phase 1: ${elapsed}h elapsed, fewest evals $evals"
    if ge "$elapsed" "$PHASE1_HOURS" && ((evals >= PHASE1_MIN_EVALS)); then break; fi
    if ge "$elapsed" "$(awk -v h="$PHASE1_HOURS" 'BEGIN { print h * 1.5 }')"; then
      log "phase 1 hard cap reached with $evals evals; deciding anyway"
      break
    fi
    sleep "$CHECK_EVERY"
  done

  compare_runs | tee "$T_DIR/compare.txt"
  compare_runs --json >"$T_DIR/compare.json" || true
  winner=$(uv run python -c 'import json,sys; print(json.load(open(sys.argv[1]))["winner"] or "")' \
    "$T_DIR/compare.json")
  [[ -n "$winner" ]] || die "factorio-compare found no winner; see $T_DIR/compare.txt"
  winner=$(basename "$winner")
  echo "{\"winner\": \"$winner\", \"decided\": \"$(date -Is)\"}" >"$T_DIR/winner.json"
  log "winner: $winner"
  state_set winner "$winner"
  for algo in "${ALGOS[@]}"; do stop_and_wait "$algo-$STAMP"; done
  state_set phase 2
  state_set phase2_start "$(date +%s)"
fi

winner=$(state_get winner)
algo=${winner%%-*}
ensure_cluster 0 $((FINAL_NUM_ENVS + 1))
log "phase 2: $winner on $FINAL_NUM_ENVS envs for ${PHASE2_HOURS}h"
while true; do
  if [[ -f "runs/$winner/checkpoint-final.pt" ]]; then
    log "$winner reached its total steps"
    break
  fi
  launch "$algo" "$winner" 0 "$FINAL_NUM_ENVS"
  snapshot
  elapsed=$(hours_since "$(state_get phase2_start)")
  log "phase 2: ${elapsed}h elapsed"
  if ge "$elapsed" "$PHASE2_HOURS"; then
    log "phase 2 time is up"
    [[ "${STOP_AFTER:-0}" == 1 ]] && stop_and_wait "$winner"
    break
  fi
  sleep "$CHECK_EVERY"
done
state_set phase done
uv run --group train factorio-monitor "runs/$winner" | tee -a "$T_DIR/tournament.log"
log "tournament finished"
