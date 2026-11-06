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
| `factorio_agent.agents` | `RandomAgent`, `ScriptedAgent`, `PolicyAgent` (untrained PyTorch actor-critic MLP) |
| `factorio_agent.env` | `make_env`, `run_episode`, `check_cluster` around FLE's gym registry |
| `notebooks/explore_agent.py` | marimo notebook for browsing tasks, inspecting the policy, and running episodes |

## Setup

Requirements: [uv](https://docs.astral.sh/uv/) and Docker ([Docker Desktop](https://www.docker.com/products/docker-desktop/) or [OrbStack](https://orbstack.dev)).
The Factorio headless server runs in Docker. On Apple Silicon FLE runs it under the `box64` emulator.

```bash
uv sync                       # creates .venv with Python 3.13 and installs everything
uv run pytest                 # offline tests, no server needed
```

Python is pinned to 3.13, the newest version FLE supports.

## Run the demo agent

```bash
uv run fle cluster start      # start 1 Factorio server (RCON on tcp/27000)
uv run factorio-agent --list  # list task ids
uv run factorio-agent --env iron_ore_throughput --agent policy --steps 10
uv run factorio-agent --agent scripted --steps 8
uv run fle cluster stop
```

Agents: `random` (uniform macro-actions), `scripted` (fixed plan: gather coal and ore, place a
fuelled burner drill, ...), and `policy` (samples from a randomly initialised PyTorch network).

To use a server that isn't a local container, set `FACTORIO_SERVER_ADDRESS` and `FACTORIO_SERVER_PORT`.

## Notebook

```bash
uv run marimo edit notebooks/explore_agent.py
```

## Notes

- FLE 0.4.3 (latest on PyPI) uses the legacy `gym` package rather than `gymnasium`.
  `reset()` returns `(obs, info)` and `step(Action(code=...))` returns the 5-tuple.
- `a2a-sdk` is pinned `<1` because FLE 0.4.3 imports an API that 1.x removed.
