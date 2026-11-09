#!/usr/bin/env bash
# Train Rainbow DQN on FLE. Extra arguments are passed to `factorio-train dqn` (they override the
# defaults below); see `uv run factorio-train dqn --help`. Environment variables: scripts/lib.sh.
#
#   scripts/train_dqn.sh
#   NUM_ENVS=4 DETACH=1 scripts/train_dqn.sh --n-step 5
#   scripts/train_dqn.sh --no-prioritized             # leave-one-out ablation

source "$(dirname "$0")/lib.sh"
NUM_ENVS=${NUM_ENVS:-8}

# FLE throughput rewards reach ~15+ per step, so returns can reach several hundred: with the
# default support of [-10, 10], a scale of 0.01 covers +-1000. Check edge_mass in the run's metrics.
#
# one iteration = one step in each env, followed by --gradient-steps updates; env steps are
# the bottleneck, so the GPU does several updates per step (replay ratio ~1 with 8 envs)
train dqn \
  --total-steps "${TOTAL_STEPS:-100000}" \
  --buffer-size 200000 \
  --batch-size 256 \
  --learning-starts 1000 \
  --gradient-steps 8 \
  --n-step 3 \
  --target-period 1000 \
  --reward-scale 0.01 \
  --hidden 256 \
  --lr 2.5e-4 \
  --checkpoint-every 2000 \
  --log-every 10 \
  "$@"
