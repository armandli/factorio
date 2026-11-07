#!/usr/bin/env bash
# Train DQN on FLE. Extra arguments are passed to `factorio-train dqn` (they override the
# defaults below); see `uv run factorio-train dqn --help`. Environment variables: scripts/lib.sh.
#
#   scripts/train_dqn.sh
#   NUM_ENVS=4 DETACH=1 scripts/train_dqn.sh --n-step 5

source "$(dirname "$0")/lib.sh"
NUM_ENVS=${NUM_ENVS:-8}

# one iteration = one step in each env, followed by --gradient-steps updates; env steps are
# the bottleneck, so the GPU does several updates per step (replay ratio ~1 with 8 envs)
train dqn \
  --total-steps "${TOTAL_STEPS:-100000}" \
  --buffer-size 200000 \
  --batch-size 256 \
  --learning-starts 1000 \
  --gradient-steps 8 \
  --n-step 3 \
  --tau 0.005 \
  --eps-fraction 0.3 \
  --hidden 256 \
  --lr 5e-4 \
  --checkpoint-every 2000 \
  --log-every 10 \
  "$@"
