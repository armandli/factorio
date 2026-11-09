"""Rainbow DQN: double Q, prioritized replay, dueling, n-step, distributional and noisy nets.

Hessel et al. (2018). Three decisions combine the components: the C51 target uses the n-step
return, the online net picks the bootstrap action for it (double Q), and the KL loss sets the
replay priorities. Each component can be left out for ablation runs: --no-double,
--no-prioritized, --no-dueling, --n-step 1, --no-distributional, --no-noisy (epsilon-greedy).

The support must fit the returns. The paper's [-10, 10] relies on rewards clipped to [-1, 1],
whereas FLE rewards are unclipped, so `reward_scale` brings them into range instead. Watch
`edge_mass`, the share of target probability on the outermost atoms: if it stays high, returns
are being clipped, so lower --reward-scale or widen --v-min/--v-max.
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
from factorio_agent.rl.buffers import NStepAccumulator, PrioritizedReplayBuffer, ReplayBuffer
from factorio_agent.rl.common import Amp, TrainConfig, Trainer
from factorio_agent.rl.envs import VecMacroEnv
from factorio_agent.rl.networks import QNetwork, categorical_projection


@dataclass
class DQNConfig(TrainConfig):
    lr: float = 2.5e-4
    adam_eps: float = 1.5e-4
    max_grad_norm: float = 10.0
    buffer_size: int = 50_000
    batch_size: int = 64
    learning_starts: int = 200  # env steps of random play before updates
    train_freq: int = 1  # env steps (per env) between updates
    gradient_steps: int = 2  # gradient steps per update
    target_period: int = 500  # env steps between hard copies to the target network
    hidden: int = 128
    # double Q-learning
    double: bool = True
    # dueling network
    dueling: bool = True
    # multi-step returns
    n_step: int = 3
    # prioritized replay
    prioritized: bool = True
    omega: float = 0.5  # priority exponent
    beta_start: float = 0.4  # importance sampling exponent, annealed to 1 over total_steps
    # distributional RL (C51)
    distributional: bool = True
    atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    reward_scale: float = 0.02  # rewards are multiplied by this before they reach the agent
    # noisy nets
    noisy: bool = True
    sigma0: float = 0.1  # the paper's 0.5 swamps C51's small action gaps at this data budget
    # epsilon-greedy, only used with --no-noisy
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_fraction: float = 0.3  # fraction of total_steps over which epsilon decays


class QAgent(Agent):
    """Greedy over a QNetwork's mean weights (eval mode turns noise off), or epsilon-greedy."""

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
        self.q = QNetwork(
            hidden=cfg.hidden,
            distributional=cfg.distributional,
            atoms=cfg.atoms,
            v_min=cfg.v_min,
            v_max=cfg.v_max,
            dueling=cfg.dueling,
            noisy=cfg.noisy,
            sigma0=cfg.sigma0,
        ).to(self.device)
        self.q_target = copy.deepcopy(self.q).requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=cfg.lr, eps=cfg.adam_eps)
        self.amp = Amp(self.device, cfg.amp)
        if envs is not None:
            fields = {
                "obs": (FEATURE_DIM,),
                "action": (),
                "reward": (),
                "next_obs": (FEATURE_DIM,),
                "done": (),
                "discount": (),
            }
            self.buffer = (
                PrioritizedReplayBuffer(cfg.buffer_size, cfg.omega, **fields)
                if cfg.prioritized
                else ReplayBuffer(cfg.buffer_size, **fields)
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
        if cfg.noisy:
            return 0.0  # exploration comes from the noise
        frac = min(self.global_step / max(cfg.eps_fraction * cfg.total_steps, 1), 1.0)
        return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)

    def beta(self) -> float:
        cfg = self.cfg
        frac = min(self.global_step / max(cfg.total_steps, 1), 1.0)
        return cfg.beta_start + frac * (1.0 - cfg.beta_start)

    @torch.no_grad()
    def _select(self, obs: torch.Tensor) -> list[int]:
        eps = self.epsilon() if self.global_step >= self.cfg.learning_starts else 1.0
        self.q.reset_noise()
        greedy = self.q(obs.to(self.device)).argmax(-1).tolist()
        return [random.randrange(NUM_ACTIONS) if random.random() < eps else a for a in greedy]

    def train_iteration(self) -> dict[str, float]:
        cfg = self.cfg
        self.q.train()  # noise on; make_agent() leaves the network in eval mode
        start_step = self.global_step
        metrics: dict[str, float] = {}
        for _ in range(cfg.train_freq):
            actions = self._select(self.obs)
            step = self.envs.step(actions)
            for i, acc in enumerate(self.nstep):
                ended = bool(step.terminated[i] or step.truncated[i])
                for tr in acc.push(
                    self.obs[i],
                    actions[i],
                    float(step.reward[i]) * cfg.reward_scale,
                    step.next_obs[i],
                    bool(step.terminated[i]),
                    ended,
                ):
                    self.buffer.add(**tr)
            self.obs = step.obs
            self.global_step += self.envs.num_envs

        if self.global_step >= cfg.learning_starts and len(self.buffer) >= cfg.batch_size:
            updates = [self._gradient_step() for _ in range(cfg.gradient_steps)]
            for key in updates[0]:
                metrics[key] = sum(u[key] for u in updates) / len(updates)
        # counted in env steps, so that resuming needs no extra state
        if self.global_step // cfg.target_period != start_step // cfg.target_period:
            self.q_target.load_state_dict(self.q.state_dict())
        metrics["epsilon"] = self.epsilon()
        if cfg.prioritized:
            metrics["beta"] = self.beta()
        return metrics

    def _gradient_step(self) -> dict[str, float]:
        cfg = self.cfg
        if cfg.prioritized:
            b = self.buffer.sample(cfg.batch_size, self.device, beta=self.beta())
            weights = b["weight"]
        else:
            b = self.buffer.sample(cfg.batch_size, self.device)
            weights = torch.ones(cfg.batch_size, device=self.device)
        rows = torch.arange(cfg.batch_size, device=self.device)
        discount = b["discount"] * (1.0 - b["done"])

        # each network pass below sees independent noise
        with torch.no_grad():
            self.q.reset_noise()
            self.q_target.reset_noise()
            # double Q: the online net picks the bootstrap action, the target net evaluates it
            selector = self.q if cfg.double else self.q_target
            next_a = selector(b["next_obs"]).argmax(-1)
            next_logits = self.q_target.logits(b["next_obs"])[rows, next_a]
        self.q.reset_noise()
        with self.amp.autocast():
            logits = self.q.logits(b["obs"])
        logits = logits.float()[rows, b["action"]]

        extra: dict[str, float] = {}
        if cfg.distributional:
            with torch.no_grad():
                target = categorical_projection(
                    next_logits.softmax(-1), b["reward"], discount, self.q.support
                )
            log_p = logits.log_softmax(-1)
            # Cross-entropy equals KL(target || p) up to the target's entropy, which has no
            # gradient. Priorities use the true KL, so a fully learned transition falls towards
            # zero even when its target distribution is wide.
            per_sample = -(target * log_p).sum(-1)
            entropy = -(target * target.clamp_min(1e-8).log()).sum(-1)
            error = (per_sample.detach() - entropy).clamp_min(0.0)
            q = (log_p.detach().exp() * self.q.support).sum(-1)
            extra["edge_mass"] = (target[:, 0] + target[:, -1]).mean().item()
        else:
            q = logits.squeeze(-1)
            target_q = b["reward"] + discount * next_logits.squeeze(-1)
            per_sample = F.smooth_l1_loss(q, target_q, reduction="none")
            error = (q.detach() - target_q).abs()
            q = q.detach()

        grad_norm = self.amp.step(
            (weights * per_sample).mean(), self.optimizer, self.q.parameters(), cfg.max_grad_norm
        )
        if cfg.prioritized:
            self.buffer.update_priorities(b["index"], error)
        return {
            "loss": error.mean().item(),  # KL if distributional, else |TD error|
            "q_mean": q.mean().item() / cfg.reward_scale,  # in the env's reward units
            "grad_norm": grad_norm,
            **extra,
        }
