#!/usr/bin/env bash
# Train the model-based MCGS agent on FLE. Extra arguments are passed to
# `factorio-train mcgs` (they override the defaults below); see
# `uv run factorio-train mcgs --help`. Environment variables: scripts/lib.sh.
#
#   scripts/train_mcgs.sh
#   NUM_ENVS=4 DETACH=1 scripts/train_mcgs.sh --num-simulations 128

source "$(dirname "$0")/lib.sh"
NUM_ENVS=${NUM_ENVS:-8}

# each env step runs a --num-simulations search per env (tiny batch-1 forwards, so the GPU
# mostly helps the --gradient-steps updates); still far cheaper than the FLE step itself
train mcgs \
  --total-steps "${TOTAL_STEPS:-50000}" \
  --num-simulations 64 \
  --buffer-size 200000 \
  --batch-size 256 \
  --learning-starts 1000 \
  --gradient-steps 2 \
  --hidden 256 \
  --lr 5e-4 \
  --checkpoint-every 1000 \
  --log-every 10 \
  "$@"
