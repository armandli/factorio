#!/usr/bin/env bash
# Train PPO on FLE. Extra arguments are passed to `factorio-train ppo` (they override the
# defaults below); see `uv run factorio-train ppo --help`. Environment variables: scripts/lib.sh.
#
#   scripts/train_ppo.sh
#   NUM_ENVS=16 TOTAL_STEPS=500000 DETACH=1 scripts/train_ppo.sh --entropy-coef 0.02
#   RUN_NAME=ppo-20261106-120000 scripts/train_ppo.sh     # resume an earlier run

source "$(dirname "$0")/lib.sh"
NUM_ENVS=${NUM_ENVS:-8}

# 8 envs x 32 steps = 256 transitions per update, ~5 min of FLE time at ~0.1 steps/s/env
train ppo \
  --total-steps "${TOTAL_STEPS:-200000}" \
  --num-steps 32 \
  --minibatch-size 64 \
  --update-epochs 4 \
  --hidden 128 \
  --lr 3e-4 \
  --entropy-coef 0.01 \
  --target-kl 0.03 \
  --checkpoint-every 2048 \
  --log-every 1 \
  "$@"
