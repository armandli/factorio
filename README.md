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

On Windows/WSL 2 with Docker Desktop, enable *Settings → Resources → WSL Integration* for your
distro (then `wsl --shutdown` and reopen). Without it, `docker` in WSL is a stub that only
prints "The command 'docker' could not be found in this WSL 2 distro".

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
uv run factorio-agent --checkpoint runs/<run>/checkpoint-best.pt   # play a trained agent
```

Each run writes `runs/<algo>-<time>/` with `config.json`, `metrics.jsonl` and checkpoints.
`--resume <checkpoint>` continues training (hyperparameters come from the command line).

### Is it improving? Evaluation, baselines and the best checkpoint

Training returns come from the *exploring* policy (PPO samples, DQN uses noisy layers, MCGS
adds search noise), so they are noisy and lag behind what the agent can actually do. With
`--eval-every N` (on by default in `scripts/`, `EVAL_EVERY=4000`):

- every N env steps the current weights are snapshotted and played for `--eval-episodes`
  episodes without exploration, on their own env (container `FIRST_CONTAINER + NUM_ENVS`, or
  `--eval-container`), in a background thread, so training doesn't wait;
- the random and scripted agents are played once per run on the same env as baselines
  (`baselines.json`);
- the snapshot with the best mean eval return is saved as `checkpoint-best.pt` (`best.json`),
  and the final weights get one last evaluation.

Results go to the console, `metrics.jsonl` (`eval_return`, `eval_vs_scripted`, ...),
TensorBoard (`eval/vs_baselines` overlays policy and baselines), and `factorio-monitor`, which
shows eval/best/baseline columns, the eval history, whether the best beats the scripted agent,
and warns when 5 evaluations pass without a new best. `factorio-eval` compares any checkpoints
against the baselines afterwards:

```bash
uv run factorio-train dqn --mock --total-steps 3000 --eval-every 500       # try it offline
uv run factorio-eval runs/ppo-*/checkpoint-best.pt runs/dqn-*/checkpoint-best.pt --container 0
```

Each run directory then holds:

| File | Contents |
| --- | --- |
| `checkpoint-<step>.pt` | periodic (last `--keep-checkpoints`), with replay buffers, for resuming |
| `checkpoint-final.pt` | end of training |
| `checkpoint-best.pt`, `best.json` | best evaluated weights (no replay buffer) and their scores |
| `baselines.json` | random / scripted baseline returns |
| `config.json`, `metrics.jsonl`, `tb/` | hyperparameters, per-iteration metrics, TensorBoard |

| Algorithm | Module | Notes |
| --- | --- | --- |
| `ppo` | `rl/ppo.py` | Clipped PPO + GAE, bootstraps truncated episodes; trains `PolicyNet` |
| `dqn` | `rl/dqn.py` | Rainbow: double Q, prioritized replay, dueling, n-step, C51, noisy nets (see below) |
| `mcgs` | `rl/mcgs.py`, `rl/search.py` | Model-based: Monte-Carlo **Graph** Search over a learned world model (AlphaZero-style policy/value targets) |

MCGS is MCTS with a transposition table. States reached by different action orders share one
node and its value, which suits Factorio, where many macro-actions commute. The world model
predicts the next *feature vector* (not a latent), so states can be matched by rounding. Search
takes any `SearchModel`, so a simulator-backed model (FLE game-state snapshots) could replace
the learned one.

`dqn` runs the full Rainbow agent by default. For leave-one-out ablation runs, use
`--no-double`, `--no-prioritized`, `--no-dueling`, `--n-step 1`, `--no-distributional` (scalar
Q-values and smooth L1 loss) or `--no-noisy` (epsilon-greedy). The `--no-distributional`,
`--no-noisy` and `--no-dueling` flags, `--atoms` and `--hidden` change the network, so they can't
be changed on `--resume`. Changing `--v-min`/`--v-max` on resume has no effect: the checkpoint's
support is kept. Don't change `--reward-scale` on resume either: the buffer holds rewards at the
old scale.

The return support has to fit the returns. FLE rewards aren't clipped as in the paper, so
`--reward-scale` (default 0.02, and 0.01 in `scripts/train_dqn.sh`) scales them to fit
`--v-min`/`--v-max` (default -10 to 10). At 0.02 that covers raw returns of ±500. Watch
`edge_mass` in `metrics.jsonl`, the share of target probability on the outermost atoms: if it
stays high, returns are being clipped, so lower the scale. `q_mean` is reported in unscaled reward
units, and `loss` is the KL divergence (|TD error| with `--no-distributional`).

`--sigma0` defaults to 0.1, not the paper's 0.5. C51's action gaps are small compared with its
atom spacing, and at 0.5 the noise swamped them on the mock env: the policy kept repeating an
arbitrary action. 0.1 helps but doesn't cure this: over 10 held-out seeds at 10k mock steps, full
Rainbow averaged 162 with 5 seeds below the scripted baseline (155), while `--no-noisy` averaged
234 with every seed at 232 or above, and the previous DQN averaged 229. The mock needs no deep
exploration, so this doesn't settle what's best on real FLE, but if a run gets stuck, try
`--no-noisy` first. Noise is off in evaluation, so evals use the mean weights.

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
scripts/train_all.sh                      # PPO + DQN + MCGS in the background, 4+1 containers each
RUN_NAME=ppo-20261106-120000 scripts/train_ppo.sh                  # resume a stopped run
scripts/stop.sh ppo-20261106-120000       # or --all; the trainer checkpoints before exiting

scripts/monitor.sh                        # dashboard: GPU, containers, processes, run progress
scripts/tensorboard.sh                    # then: ssh -N -L 6006:127.0.0.1:6006 <gpu-host>
uv run factorio-monitor runs/<run>        # one run: metrics, return curve, eval vs baselines
```

The train scripts install the `train` dependency group (TensorBoard), check for CUDA, and start
enough Factorio containers. If `factorio-train` crashes (for example, a server stops
responding), they restart that run's containers and resume from the newest checkpoint, up to
`MAX_RESTARTS` times. Checkpoints include the replay buffer, so DQN and MCGS resume exactly.
Settings are environment variables (`NUM_ENVS`, `TOTAL_STEPS`, `FIRST_CONTAINER`, `ENV_ID`,
`ERROR_PENALTY`, `EVAL_EVERY`, `EVAL_EPISODES`, ...; see `scripts/lib.sh`), plus any
`factorio-train` flag as an argument. Each run uses `NUM_ENVS` containers plus one for
evaluation (`EVAL_EVERY=0` turns it off). Runs that share a machine need disjoint container
ranges (`FIRST_CONTAINER`); `train_all.sh` handles this. Budget about 1 CPU core and 1 GB of RAM per container. The networks are small
and FLE steps are the bottleneck, so the GPU mostly speeds up DQN and MCGS gradient updates.

## Notebook

```bash
uv run marimo edit notebooks/explore_agent.py
```

## Notes

- FLE 0.4.3 (latest on PyPI) uses the legacy `gym` package rather than `gymnasium`.
  `reset()` returns `(obs, info)` and `step(Action(code=...))` returns the 5-tuple.
- `a2a-sdk` is pinned `<1` because FLE 0.4.3 imports an API that 1.x removed.
