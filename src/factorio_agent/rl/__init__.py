"""RL training for FLE: PPO, DQN and model-based MCGS on a shared env/trainer layer.

Add an algorithm by subclassing `Trainer` (and `TrainConfig` for its hyperparameters) and
registering it in ALGORITHMS; `factorio-train <name>` then picks it up with a flag per field.
"""

from pathlib import Path

import torch

from factorio_agent.agents import Agent
from factorio_agent.rl.common import TrainConfig, Trainer
from factorio_agent.rl.dqn import DQNConfig, DQNTrainer
from factorio_agent.rl.envs import MacroEnv, VecMacroEnv, make_eval_env, make_vec_env
from factorio_agent.rl.evaluate import EvalResult, Evaluator, evaluate
from factorio_agent.rl.mcgs import MCGSConfig, MCGSTrainer
from factorio_agent.rl.ppo import PPOConfig, PPOTrainer

ALGORITHMS: dict[str, type[Trainer]] = {t.name: t for t in (PPOTrainer, DQNTrainer, MCGSTrainer)}

__all__ = [
    "ALGORITHMS",
    "DQNConfig",
    "DQNTrainer",
    "EvalResult",
    "Evaluator",
    "MCGSConfig",
    "MCGSTrainer",
    "MacroEnv",
    "PPOConfig",
    "PPOTrainer",
    "TrainConfig",
    "Trainer",
    "VecMacroEnv",
    "evaluate",
    "load_agent",
    "load_trainer",
    "make_eval_env",
    "make_vec_env",
]


def load_trainer(path: str | Path, envs: VecMacroEnv | None = None, **overrides) -> Trainer:
    """Rebuild a trainer from a checkpoint (pass `envs` to resume training)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cls = ALGORITHMS[ckpt["algo"]]
    cfg = cls.config_cls(**{**ckpt["config"], **overrides})
    trainer = cls(cfg, envs)
    trainer.load(ckpt)
    return trainer


def load_agent(path: str | Path, device: str = "cpu") -> Agent:
    """An `Agent` playing with a checkpoint's weights, usable with `env.run_episode`."""
    return load_trainer(path, device=device).make_agent()
