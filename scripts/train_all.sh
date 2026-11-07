#!/usr/bin/env bash
# Start PPO, DQN and MCGS side by side in the background, each on its own range of Factorio
# containers (NUM_ENVS each, default 4, plus one evaluation container unless EVAL_EVERY=0, so
# 15 containers; budget ~1 CPU core and ~1 GB RAM per container). Extra arguments are passed
# to all three.
#
#   scripts/train_all.sh
#   NUM_ENVS=8 scripts/train_all.sh

source "$(dirname "$0")/lib.sh"
export NUM_ENVS=${NUM_ENVS:-4}
algos=(ppo dqn mcgs)
stamp=$(date +%Y%m%d-%H%M%S)

# start every container once, up front, so the three runs don't race to create the cluster
setup_python
per_run=$(run_containers)
ensure_cluster 0 $((${#algos[@]} * per_run))

for i in "${!algos[@]}"; do
  algo=${algos[$i]}
  FIRST_CONTAINER=$((i * per_run)) RUN_NAME="$algo-$stamp" DETACH=1 \
    "$REPO_ROOT/scripts/train_$algo.sh" "$@"
done
echo
echo "monitor: scripts/monitor.sh    tensorboard: scripts/tensorboard.sh"
