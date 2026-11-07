# factorio

Agents that play the [Factorio Learning Environment](https://github.com/JackHopkins/factorio-learning-environment) (FLE).
Models are written in PyTorch; research notebooks use [marimo](https://marimo.io).

## How FLE works

An FLE action is a **Python program** run in the game's REPL, using tools like `nearest`,
`move_to`, `harvest_resource`, `place_entity` and `craft_item`. The observation is a dict
holding the program's output (`raw_text`), inventory, entities, score, flows and so on.

This repo turns that into a standard discrete RL problem:

| Module | Purpose |
| --- | --- |
| `factorio_agent.actions` | `MACRO_ACTIONS`: a fixed library of code snippets (harvest ore, place a drill, craft, wait, ...) |
| `factorio_agent.features` | `featurize(obs)`: observation dict to a fixed-size `torch.Tensor` |
| `factorio_agent.agents` | `RandomAgent`, `ScriptedAgent`, `PolicyAgent` (PyTorch actor-critic MLP) |
| `factorio_agent.env` | `make_env`, `run_episode`, `check_cluster` around FLE's gym registry |
| `factorio_agent.rl` | Training: PPO, DQN and model-based MCGS (see below) |
| `notebooks/explore_agent.py` | marimo notebook for browsing tasks, inspecting the policy, and running episodes |

## Setup

Requirements: [uv](https://docs.astral.sh/uv/) and Docker ([Docker Desktop](https://www.docker.com/products/docker-desktop/) or [OrbStack](https://orbstack.dev)).
The Factorio headless server runs in Docker. On Apple Silicon FLE runs it under the `box64` emulator.

```bash
uv sync                       # creates .venv with Python 3.13 and installs everything
uv run pytest                 # offline tests, no server needed
```

Python is pinned to 3.13, the newest version FLE supports.

With [Colima](https://github.com/abiosoft/colima) instead of Docker Desktop:

```bash
brew install docker colima
colima start --cpu 4 --memory 8 --vm-type vz --vz-rosetta
```

If image pulls fail with `docker-credential-desktop: executable file not found`, remove the
`"credsStore": "desktop"` line from `~/.docker/config.json` (left over from Docker Desktop).
`fle cluster start` still reports success when the pull fails, so check `docker ps`.

## Run the demo agent

```bash
uv run fle cluster start      # start 1 Factorio server (RCON on tcp/27000)
uv run factorio-agent --list  # list task ids
uv run factorio-agent --env iron_ore_throughput --agent policy --steps 10
uv run factorio-agent --agent scripted --steps 8
uv run fle cluster stop
```

Agents: `random` (uniform macro-actions), `scripted` (fixed plan: gather coal and ore, place a
fuelled burner drill, ...), and `policy` (samples from a randomly initialised PyTorch network; use `--checkpoint` for a trained one).

To use a server that isn't a local container, set `FACTORIO_SERVER_ADDRESS` and `FACTORIO_SERVER_PORT`.

## Training

```bash
uv run factorio-train ppo --mock --total-steps 20000 --num-envs 4  # offline mock env, seconds
uv run factorio-train dqn --total-steps 2000                       # real FLE server
uv run factorio-train mcgs --num-envs 2 --num-simulations 64       # needs `fle cluster start -n 2`
uv run factorio-train ppo --help                                   # every hyperparameter is a flag
uv run factorio-agent --checkpoint runs/<run>/checkpoint-final.pt  # play a trained agent
```

Each run writes `runs/<algo>-<time>/` with `config.json`, `metrics.jsonl` and checkpoints.
`--resume <checkpoint>` continues training (hyperparameters come from the command line).

| Algorithm | Module | Notes |
| --- | --- | --- |
| `ppo` | `rl/ppo.py` | Clipped PPO + GAE, bootstraps truncated episodes; trains `PolicyNet` |
| `dqn` | `rl/dqn.py` | Double DQN, dueling head, n-step returns, Polyak target |
| `mcgs` | `rl/mcgs.py`, `rl/search.py` | Model-based: Monte-Carlo **Graph** Search over a learned world model (AlphaZero-style policy/value targets) |

MCGS is MCTS with a transposition table. States reached by different action orders share one
node and its value, which suits Factorio, where many macro-actions commute. The world model
predicts the next *feature vector* (not a latent), so states can be matched by rounding. Search
takes any `SearchModel`, so a simulator-backed model (FLE game-state snapshots) could replace
the learned one.

Shared pieces: `rl/envs.py` (`MacroEnv`: int actions to featurized tensors, step limit, error
penalty; `VecMacroEnv`: threaded envs, one Factorio container each, auto-reset),
`rl/buffers.py`, and `rl/common.py` (`Trainer` base, config, AMP, logging, checkpoints). To add
an algorithm, subclass `Trainer` and `TrainConfig` and register it in `rl/__init__.py`.

`rl/mock_env.py` is a toy offline stand-in that speaks FLE's API (drills produce ore each step),
for developing and testing training code. Real FLE runs at about 0.1 steps/sec per container, so
use `--mock` to iterate and add containers (`--num-envs`) to scale up.

### On a GPU machine

`scripts/` wraps `factorio-train` for long unattended runs on a Linux box with an NVIDIA GPU,
Docker and uv:

```bash
scripts/train_ppo.sh                      # foreground; also train_dqn.sh, train_mcgs.sh
DETACH=1 NUM_ENVS=16 scripts/train_mcgs.sh --num-simulations 128   # background, extra flags
scripts/train_all.sh                      # PPO + DQN + MCGS in the background, 4 containers each
RUN_NAME=ppo-20261106-120000 scripts/train_ppo.sh                  # resume a stopped run
scripts/stop.sh ppo-20261106-120000       # or --all; the trainer checkpoints before exiting

scripts/monitor.sh                        # dashboard: GPU, containers, processes, run progress
scripts/tensorboard.sh                    # then: ssh -N -L 6006:127.0.0.1:6006 <gpu-host>
uv run factorio-monitor runs/<run>        # one run: latest metrics and return curve
```

The train scripts install the `train` dependency group (TensorBoard), check for CUDA, and start
enough Factorio containers. If `factorio-train` crashes (for example, a server stops
responding), they restart that run's containers and resume from the newest checkpoint, up to
`MAX_RESTARTS` times. Checkpoints include the replay buffer, so DQN and MCGS resume exactly.
Settings are environment variables (`NUM_ENVS`, `TOTAL_STEPS`, `FIRST_CONTAINER`, `ENV_ID`,
`ERROR_PENALTY`, ...; see `scripts/lib.sh`), plus any `factorio-train` flag as an argument.
Runs that share a machine need disjoint container ranges (`FIRST_CONTAINER`); `train_all.sh`
handles this. Budget about 1 CPU core and 1 GB of RAM per container. The networks are small
and FLE steps are the bottleneck, so the GPU mostly speeds up DQN and MCGS gradient updates.

## Notebook

```bash
uv run marimo edit notebooks/explore_agent.py
```

## Notes

- FLE 0.4.3 (latest on PyPI) uses the legacy `gym` package rather than `gymnasium`.
  `reset()` returns `(obs, info)` and `step(Action(code=...))` returns the 5-tuple.
- `a2a-sdk` is pinned `<1` because FLE 0.4.3 imports an API that 1.x removed.
