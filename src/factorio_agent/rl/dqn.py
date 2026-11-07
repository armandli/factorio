"""Deep Q-learning: double DQN with a dueling head, n-step returns and a Polyak target net.

FLE steps take seconds, so the defaults lean sample-efficient: n-step returns, several
gradient steps per env step, and a short warm-up.
"""

import copy
import random
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from factorio_agent.actions import MACRO_ACTIONS, NUM_ACTIONS, MacroAction
from factorio_agent.agents import Agent
from factorio_agent.features import FEATURE_DIM, featurize
from factorio_agent.rl.buffers import NStepAccumulator, ReplayBuffer
from factorio_agent.rl.common import Amp, TrainConfig, Trainer
from factorio_agent.rl.envs import VecMacroEnv
from factorio_agent.rl.networks import QNetwork


@dataclass
class DQNConfig(TrainConfig):
    lr: float = 1e-3
    buffer_size: int = 50_000
    batch_size: int = 64
    learning_starts: int = 200  # env steps of random play before updates
    train_freq: int = 1  # env steps (per env) between updates
    gradient_steps: int = 2  # gradient steps per update
    n_step: int = 3
    tau: float = 0.01  # Polyak rate for the target network
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_fraction: float = 0.3  # fraction of total_steps over which epsilon decays
    hidden: int = 128


class QAgent(Agent):
    """Epsilon-greedy over a QNetwork (epsilon=0: greedy)."""

    def __init__(self, net: QNetwork, epsilon: float = 0.0, device: str = "cpu"):
        self.net = net.eval()
        self.epsilon = epsilon
        self.device = torch.device(device)

    @torch.no_grad()
    def act(self, obs: dict[str, Any]) -> MacroAction:
        if random.random() < self.epsilon:
            return random.choice(MACRO_ACTIONS)
        q = self.net(featurize(obs).to(self.device))
        return MACRO_ACTIONS[int(q.argmax())]


class DQNTrainer(Trainer):
    name = "dqn"
    config_cls = DQNConfig
    cfg: DQNConfig

    def __init__(self, cfg: DQNConfig, envs: VecMacroEnv | None):
        super().__init__(cfg, envs)
        self.q = QNetwork(hidden=cfg.hidden).to(self.device)
        self.q_target = copy.deepcopy(self.q).requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.amp = Amp(self.device, cfg.amp)
        if envs is not None:
            self.buffer = ReplayBuffer(
                cfg.buffer_size,
                obs=(FEATURE_DIM,),
                action=(),
                reward=(),
                next_obs=(FEATURE_DIM,),
                done=(),
                discount=(),
            )
            self.nstep = [NStepAccumulator(cfg.n_step, cfg.gamma) for _ in range(envs.num_envs)]
            self.obs = envs.reset()

    def modules(self) -> dict[str, Any]:
        modules = {"q": self.q, "q_target": self.q_target, "optimizer": self.optimizer}
        if self.envs is not None:
            modules["buffer"] = self.buffer
        return modules

    def make_agent(self) -> QAgent:
        return QAgent(self.q, device=str(self.device))

    def epsilon(self) -> float:
        cfg = self.cfg
        frac = min(self.global_step / max(cfg.eps_fraction * cfg.total_steps, 1), 1.0)
        return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)

    @torch.no_grad()
    def _select(self, obs: torch.Tensor) -> list[int]:
        eps = self.epsilon() if self.global_step >= self.cfg.learning_starts else 1.0
        greedy = self.q(obs.to(self.device)).argmax(-1).tolist()
        return [random.randrange(NUM_ACTIONS) if random.random() < eps else a for a in greedy]

    def train_iteration(self) -> dict[str, float]:
        cfg = self.cfg
        metrics: dict[str, float] = {}
        for _ in range(cfg.train_freq):
            actions = self._select(self.obs)
            step = self.envs.step(actions)
            for i, acc in enumerate(self.nstep):
                ended = bool(step.terminated[i] or step.truncated[i])
                for tr in acc.push(
                    self.obs[i],
                    actions[i],
                    float(step.reward[i]),
                    step.next_obs[i],
                    bool(step.terminated[i]),
                    ended,
                ):
                    self.buffer.add(**tr)
            self.obs = step.obs
            self.global_step += self.envs.num_envs

        if self.global_step >= cfg.learning_starts and len(self.buffer) >= cfg.batch_size:
            losses = [self._gradient_step() for _ in range(cfg.gradient_steps)]
            metrics["loss"] = sum(l for l, _, _ in losses) / len(losses)
            metrics["q_mean"] = sum(q for _, q, _ in losses) / len(losses)
            metrics["grad_norm"] = sum(g for _, _, g in losses) / len(losses)
        metrics["epsilon"] = self.epsilon()
        return metrics

    def _gradient_step(self) -> tuple[float, float, float]:
        b = self.buffer.sample(self.cfg.batch_size, self.device)
        with torch.no_grad():
            # double DQN: online net picks the action, target net evaluates it
            next_a = self.q(b["next_obs"]).argmax(-1, keepdim=True)
            next_q = self.q_target(b["next_obs"]).gather(-1, next_a).squeeze(-1)
            target = b["reward"] + b["discount"] * (1.0 - b["done"]) * next_q
        with self.amp.autocast():
            q_all = self.q(b["obs"])
        q = q_all.float().gather(-1, b["action"].unsqueeze(-1)).squeeze(-1)
        loss = F.smooth_l1_loss(q, target)
        grad_norm = self.amp.step(loss, self.optimizer, self.q.parameters(), self.cfg.max_grad_norm)
        with torch.no_grad():
            for p, tp in zip(self.q.parameters(), self.q_target.parameters(), strict=True):
                tp.lerp_(p, self.cfg.tau)
        return loss.item(), q.mean().item(), grad_norm
