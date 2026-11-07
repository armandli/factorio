"""Model-based RL: act with MCGS over a learned world model, AlphaZero-style.

Each env step runs a graph search (factorio_agent.rl.search) from the current features,
using `WorldModel` for transitions and `PolicyNet` for priors and leaf values. The
replay buffer stores (s, a, r, s', done, search policy) and three losses are trained:

- dynamics: predict s', r and termination from (s, a);
- policy: cross-entropy towards the search's root visit distribution;
- value: TD(0) target r + gamma * V_target(s'), with a Polyak-averaged target net.

Searching is cheap (small MLPs) compared with an FLE step, so planning with more
simulations trades compute for fewer real environment steps.
"""

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from factorio_agent.actions import MACRO_ACTIONS, NUM_ACTIONS, MacroAction
from factorio_agent.agents import Agent
from factorio_agent.features import FEATURE_DIM, featurize
from factorio_agent.rl.buffers import ReplayBuffer
from factorio_agent.rl.common import Amp, TrainConfig, Trainer
from factorio_agent.rl.envs import VecMacroEnv
from factorio_agent.rl.networks import PolicyNet, WorldModel
from factorio_agent.rl.search import MCGS, LearnedModel


@dataclass
class MCGSConfig(TrainConfig):
    lr: float = 1e-3
    num_simulations: int = 32
    c_puct: float = 1.25
    key_decimals: int = 1  # rounding of features when matching transpositions
    dirichlet_alpha: float = 0.3
    dirichlet_frac: float = 0.25
    temperature: float = 1.0  # sampling temperature over root visits while training
    buffer_size: int = 50_000
    batch_size: int = 64
    learning_starts: int = 200  # env steps of random play before searching / updating
    gradient_steps: int = 4  # gradient steps per env step (per env)
    tau: float = 0.01
    model_coef: float = 1.0
    policy_coef: float = 1.0
    value_coef: float = 1.0
    hidden: int = 128


class SearchAgent(Agent):
    """Plays the most-visited root action of an MCGS search."""

    def __init__(self, search: MCGS):
        self.search = search

    def act(self, obs: dict[str, Any]) -> MacroAction:
        result = self.search.search(featurize(obs))
        return MACRO_ACTIONS[int(result.policy.argmax())]


class MCGSTrainer(Trainer):
    name = "mcgs"
    config_cls = MCGSConfig
    cfg: MCGSConfig

    def __init__(self, cfg: MCGSConfig, envs: VecMacroEnv | None):
        super().__init__(cfg, envs)
        self.prediction = PolicyNet(hidden=cfg.hidden).to(self.device)
        self.dynamics = WorldModel(hidden=cfg.hidden).to(self.device)
        self.prediction_target = copy.deepcopy(self.prediction).requires_grad_(False)
        self.params = [*self.prediction.parameters(), *self.dynamics.parameters()]
        self.optimizer = torch.optim.Adam(self.params, lr=cfg.lr)
        self.amp = Amp(self.device, cfg.amp)
        self.search = MCGS(
            LearnedModel(self.prediction, self.dynamics, self.device),
            NUM_ACTIONS,
            num_simulations=cfg.num_simulations,
            gamma=cfg.gamma,
            c_puct=cfg.c_puct,
            key_decimals=cfg.key_decimals,
            dirichlet_alpha=cfg.dirichlet_alpha,
            dirichlet_frac=cfg.dirichlet_frac,
        )
        if envs is not None:
            self.buffer = ReplayBuffer(
                cfg.buffer_size,
                obs=(FEATURE_DIM,),
                action=(),
                reward=(),
                next_obs=(FEATURE_DIM,),
                done=(),
                policy=(NUM_ACTIONS,),
            )
            self.obs = envs.reset()

    def modules(self) -> dict[str, Any]:
        modules = {
            "prediction": self.prediction,
            "dynamics": self.dynamics,
            "prediction_target": self.prediction_target,
            "optimizer": self.optimizer,
        }
        if self.envs is not None:
            modules["buffer"] = self.buffer
        return modules

    def make_agent(self) -> SearchAgent:
        return SearchAgent(self.search)

    def _act(self, obs: torch.Tensor) -> tuple[int, torch.Tensor, float, int]:
        """Returns (action, search policy, root value, graph size)."""
        if self.global_step < self.cfg.learning_starts:
            uniform = torch.full((NUM_ACTIONS,), 1.0 / NUM_ACTIONS)
            return int(torch.randint(NUM_ACTIONS, ())), uniform, 0.0, 0
        self.prediction.eval()
        self.dynamics.eval()
        result = self.search.search(obs, add_noise=True)
        if self.cfg.temperature > 0:
            probs = result.policy ** (1.0 / self.cfg.temperature)
            action = int(torch.multinomial(probs / probs.sum(), 1))
        else:
            action = int(result.policy.argmax())
        return action, result.policy, result.value, result.num_nodes

    def train_iteration(self) -> dict[str, float]:
        cfg = self.cfg
        acted = [self._act(o) for o in self.obs]
        step = self.envs.step([a for a, *_ in acted])
        for i, (action, policy, _, _) in enumerate(acted):
            self.buffer.add(
                obs=self.obs[i],
                action=action,
                reward=step.reward[i],
                next_obs=step.next_obs[i],
                done=step.terminated[i].float(),
                policy=policy,
            )
        self.obs = step.obs
        self.global_step += self.envs.num_envs

        metrics: dict[str, float] = {}
        if self.global_step >= cfg.learning_starts:
            metrics["root_value"] = sum(v for *_, v, _ in acted) / len(acted)
            metrics["graph_nodes"] = sum(n for *_, n in acted) / len(acted)
        if self.global_step >= cfg.learning_starts and len(self.buffer) >= cfg.batch_size:
            self.prediction.train()
            self.dynamics.train()
            losses = [self._gradient_step() for _ in range(cfg.gradient_steps * len(acted))]
            for k in losses[0]:
                metrics[k] = sum(m[k] for m in losses) / len(losses)
        return metrics

    def _gradient_step(self) -> dict[str, float]:
        cfg = self.cfg
        b = self.buffer.sample(cfg.batch_size, self.device)
        with torch.no_grad():
            _, next_v = self.prediction_target(b["next_obs"])
            v_target = b["reward"] + cfg.gamma * (1.0 - b["done"]) * next_v.float()
        with self.amp.autocast():
            logits, value = self.prediction(b["obs"])
            pred_next, pred_r, done_logit = self.dynamics(b["obs"], b["action"])
        policy_loss = -(b["policy"] * F.log_softmax(logits.float(), -1)).sum(-1).mean()
        value_loss = F.smooth_l1_loss(value.float(), v_target)
        obs_loss = F.mse_loss(pred_next.float(), b["next_obs"])
        reward_loss = F.mse_loss(pred_r.float(), b["reward"])
        done_loss = F.binary_cross_entropy_with_logits(done_logit.float(), b["done"])
        model_loss = obs_loss + reward_loss + done_loss
        loss = (
            cfg.policy_coef * policy_loss
            + cfg.value_coef * value_loss
            + cfg.model_coef * model_loss
        )
        grad_norm = self.amp.step(loss, self.optimizer, self.params, cfg.max_grad_norm)
        with torch.no_grad():
            for p, tp in zip(
                self.prediction.parameters(), self.prediction_target.parameters(), strict=True
            ):
                tp.lerp_(p, cfg.tau)
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "obs_loss": obs_loss.item(),
            "reward_loss": reward_loss.item(),
            "done_loss": done_loss.item(),
            "grad_norm": grad_norm,
        }
