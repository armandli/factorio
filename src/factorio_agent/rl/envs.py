"""Discrete-action, tensor-observation environments for RL training.

`MacroEnv` wraps anything with FLE's dict API (`reset(options=...) -> (obs, info)`,
`step(action) -> (obs, reward, terminated, truncated, info)`): the real FLE gym env, or
`MockFactorioEnv` for offline development. Actions are indices into MACRO_ACTIONS and
observations are `featurize`d tensors.

`VecMacroEnv` steps several of them in threads (FLE calls are I/O-bound RCON round trips, one
container per env) and auto-resets finished episodes.
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import torch

from factorio_agent.actions import MACRO_ACTIONS
from factorio_agent.features import featurize


@dataclass
class EnvStep:
    obs: torch.Tensor
    reward: float
    terminated: bool
    truncated: bool
    error: bool
    raw_obs: dict[str, Any]


class MacroEnv:
    """Single env: int action -> featurized observation, with a step limit and error penalty."""

    def __init__(
        self,
        env: Any,
        action_cls: Callable[..., Any],
        max_episode_steps: int = 50,
        error_penalty: float = 0.0,
    ):
        self.env = env
        self.action_cls = action_cls
        self.max_episode_steps = max_episode_steps
        self.error_penalty = error_penalty
        self.raw_obs: dict[str, Any] = {}
        self._t = 0

    def reset(self) -> torch.Tensor:
        self.raw_obs, _ = self.env.reset(options={"game_state": None})
        self._t = 0
        return featurize(self.raw_obs)

    def step(self, action: int) -> EnvStep:
        code = MACRO_ACTIONS[action].code
        self.raw_obs, reward, terminated, truncated, info = self.env.step(
            self.action_cls(code=code, agent_idx=0)
        )
        self._t += 1
        error = bool(info.get("error_occurred", False))
        return EnvStep(
            obs=featurize(self.raw_obs),
            reward=float(reward) - self.error_penalty * error,
            terminated=bool(terminated),
            truncated=bool(truncated) or self._t >= self.max_episode_steps,
            error=error,
            raw_obs=self.raw_obs,
        )

    def close(self) -> None:
        self.env.close()


@dataclass
class VecStep:
    """One step of N envs. `obs` is post-reset; `next_obs` is the true successor of each step
    (they differ only where an episode ended), so truncated episodes can be bootstrapped."""

    obs: torch.Tensor  # [N, D]
    reward: torch.Tensor  # [N]
    terminated: torch.Tensor  # [N] bool
    truncated: torch.Tensor  # [N] bool
    next_obs: torch.Tensor  # [N, D]
    error: torch.Tensor  # [N] bool


@dataclass
class EpisodeStats:
    episode_return: float
    length: int
    errors: int


class VecMacroEnv:
    def __init__(self, envs: list[MacroEnv]):
        self.envs = envs
        self.num_envs = len(envs)
        self._pool = ThreadPoolExecutor(max_workers=self.num_envs)
        self._returns = [0.0] * self.num_envs
        self._lengths = [0] * self.num_envs
        self._errors = [0] * self.num_envs
        self._finished: list[EpisodeStats] = []

    def reset(self) -> torch.Tensor:
        self._returns = [0.0] * self.num_envs
        self._lengths = [0] * self.num_envs
        self._errors = [0] * self.num_envs
        return torch.stack(list(self._pool.map(MacroEnv.reset, self.envs)))

    def step(self, actions: torch.Tensor | list[int]) -> VecStep:
        actions = [int(a) for a in actions]
        steps = list(self._pool.map(MacroEnv.step, self.envs, actions))
        next_obs = torch.stack([s.obs for s in steps])
        obs = next_obs.clone()
        for i, s in enumerate(steps):
            self._returns[i] += s.reward
            self._lengths[i] += 1
            self._errors[i] += s.error
            if s.terminated or s.truncated:
                self._finished.append(
                    EpisodeStats(self._returns[i], self._lengths[i], self._errors[i])
                )
                self._returns[i], self._lengths[i], self._errors[i] = 0.0, 0, 0
                obs[i] = self.envs[i].reset()
        return VecStep(
            obs=obs,
            reward=torch.tensor([s.reward for s in steps], dtype=torch.float32),
            terminated=torch.tensor([s.terminated for s in steps]),
            truncated=torch.tensor([s.truncated for s in steps]),
            next_obs=next_obs,
            error=torch.tensor([s.error for s in steps]),
        )

    def pop_episode_stats(self) -> list[EpisodeStats]:
        finished, self._finished = self._finished, []
        return finished

    def close(self) -> None:
        for env in self.envs:
            env.close()
        self._pool.shutdown()


def make_vec_env(
    env_id: str = "iron_ore_throughput",
    num_envs: int = 1,
    max_episode_steps: int = 50,
    error_penalty: float = 0.0,
    mock: bool = False,
    seed: int = 0,
    first_container: int = 0,
) -> VecMacroEnv:
    """Env i connects to Factorio container `first_container + i` (start them with
    `fle cluster start -n N`).
    With `mock=True` no server is needed."""
    if mock:
        from factorio_agent.rl.mock_env import CodeAction, MockFactorioEnv

        raw = [MockFactorioEnv(seed=seed + i) for i in range(num_envs)]
        action_cls = CodeAction
    else:
        from factorio_agent import env as fenv

        fenv.check_cluster()
        raw = [fenv.make_env(env_id, run_idx=first_container + i) for i in range(num_envs)]
        action_cls = fenv.Action
    return VecMacroEnv([MacroEnv(e, action_cls, max_episode_steps, error_penalty) for e in raw])
