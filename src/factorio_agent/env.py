"""Thin helpers around the FLE gym registry."""

import contextlib
import io
import shutil
import subprocess
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

# FLE depends on the legacy `gym` package (which prints a deprecation banner) and some
# py2-era modules that warn on import; silence both.
with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
    warnings.simplefilter("ignore")
    import gym
    from fle.env.gym_env.action import Action
    from fle.env.gym_env.registry import get_environment_info, list_available_environments

from factorio_agent.agents import Agent


def _serialise_slpp() -> None:
    """FLE parses every RCON reply with slpp's module-level `slpp` instance, whose decode keeps
    its cursor (text, at, ch) on the instance. VecMacroEnv steps envs in threads, so concurrent
    replies (e.g. the ~90 KB research state every reset fetches) corrupt each other's parse, and
    FLE raises "Could not save research state". Parsing is cheap next to a game step, so a lock
    is enough."""
    import functools
    import threading

    from slpp import SLPP

    if getattr(SLPP, "_factorio_agent_locked", False):
        return
    lock = threading.RLock()

    def locked(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            with lock:
                return method(self, *args, **kwargs)

        return wrapper

    SLPP.decode = locked(SLPP.decode)
    SLPP.encode = locked(SLPP.encode)
    SLPP._factorio_agent_locked = True


_serialise_slpp()

__all__ = [
    "Action",
    "StepRecord",
    "check_cluster",
    "get_environment_info",
    "list_available_environments",
    "make_env",
    "run_episode",
]


def check_cluster() -> list[str]:
    """Return names of running Factorio containers, or raise with a hint on how to start them."""
    if shutil.which("docker") is None:
        raise RuntimeError(
            "Docker is not installed. Install Docker Desktop or OrbStack, then run "
            "`uv run fle cluster start`."
        )
    result = subprocess.run(
        ["docker", "ps", "--filter", "name=factorio_", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Docker is not running: {result.stderr.strip()}")
    names = result.stdout.split()
    if not names:
        raise RuntimeError(
            "No Factorio containers running. Start one with `uv run fle cluster start`."
        )
    return names


def make_env(env_id: str = "iron_ore_throughput", run_idx: int = 0) -> gym.Env:
    """Create an FLE environment connected to the `run_idx`-th local Factorio container.

    Set FACTORIO_SERVER_ADDRESS / FACTORIO_SERVER_PORT to use a remote server instead.
    """
    return gym.make(env_id, run_idx=run_idx, disable_env_checker=True)


@dataclass
class StepRecord:
    step: int
    action: str
    code: str
    reward: float
    output: str
    error: bool
    ticks: int
    inventory: dict[str, int] = field(default_factory=dict)


def run_episode(
    env: gym.Env,
    agent: Agent,
    max_steps: int = 10,
    on_step: Callable[[StepRecord], None] | None = None,
) -> list[StepRecord]:
    """Roll out one episode and return per-step records."""
    obs, _ = env.reset(options={"game_state": None})
    agent.reset()
    records = []
    for t in range(max_steps):
        macro = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(Action(code=macro.code, agent_idx=0))
        record = StepRecord(
            step=t,
            action=macro.name,
            code=macro.code,
            reward=float(reward),
            output=str(obs.get("raw_text", "")),
            error=bool(info.get("error_occurred", False)),
            ticks=int(info.get("ticks", 0)),
            inventory={i["type"]: int(i["quantity"]) for i in obs.get("inventory", [])},
        )
        records.append(record)
        if on_step:
            on_step(record)
        if terminated or truncated:
            break
    return records
