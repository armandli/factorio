"""Proximal Policy Optimisation (clipped objective, GAE, value clipping)."""

from dataclasses import dataclass
from typing import Any

import torch
from torch.distributions import Categorical

from factorio_agent.agents import PolicyAgent
from factorio_agent.features import FEATURE_DIM
from factorio_agent.rl.buffers import RolloutBuffer
from factorio_agent.rl.common import Amp, TrainConfig, Trainer
from factorio_agent.rl.envs import VecMacroEnv
from factorio_agent.rl.networks import PolicyNet


@dataclass
class PPOConfig(TrainConfig):
    num_steps: int = 32  # rollout length per env
    update_epochs: int = 4
    minibatch_size: int = 32
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    target_kl: float = 0.0  # stop the epoch loop early above this approx KL (0: off)
    anneal_lr: bool = True
    hidden: int = 64


class PPOTrainer(Trainer):
    name = "ppo"
    config_cls = PPOConfig
    cfg: PPOConfig

    def __init__(self, cfg: PPOConfig, envs: VecMacroEnv | None):
        super().__init__(cfg, envs)
        self.net = PolicyNet(hidden=cfg.hidden).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, eps=1e-5)
        self.amp = Amp(self.device, cfg.amp)
        if envs is not None:
            self.buffer = RolloutBuffer(cfg.num_steps, envs.num_envs, FEATURE_DIM)
            self.obs = envs.reset()

    def modules(self) -> dict[str, Any]:
        return {"net": self.net, "optimizer": self.optimizer}

    def make_agent(self) -> PolicyAgent:
        return PolicyAgent(net=self.net, device=str(self.device))

    @torch.no_grad()
    def _policy(self, obs: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        logits, value = self.net(obs.to(self.device))
        return Categorical(logits=logits.float()), value.float()

    def train_iteration(self) -> dict[str, float]:
        cfg = self.cfg
        if cfg.anneal_lr:
            frac = 1.0 - self.global_step / cfg.total_steps
            self.optimizer.param_groups[0]["lr"] = cfg.lr * max(frac, 0.0)

        self.net.eval()
        for _ in range(cfg.num_steps):
            dist, value = self._policy(self.obs)
            action = dist.sample()
            step = self.envs.step(action)
            reward = step.reward.clone()
            # bootstrap truncated episodes from the value of their true final state
            if step.truncated.any():
                _, final_value = self._policy(step.next_obs)
                trunc = step.truncated & ~step.terminated
                reward[trunc] += cfg.gamma * final_value.cpu()[trunc]
            done = (step.terminated | step.truncated).float()
            self.buffer.add(
                self.obs, action.cpu(), dist.log_prob(action).cpu(), reward, done, value.cpu()
            )
            self.obs = step.obs
            self.global_step += self.envs.num_envs

        _, last_value = self._policy(self.obs)
        self.buffer.compute_gae(last_value.cpu(), cfg.gamma, cfg.gae_lambda)
        return self._update()

    def _update(self) -> dict[str, float]:
        cfg = self.cfg
        self.net.train()
        stats: dict[str, list[float]] = {
            k: []
            for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "grad_norm")
        }
        for _ in range(cfg.update_epochs):
            for mb in self.buffer.minibatches(cfg.minibatch_size):
                mb = {k: v.to(self.device) for k, v in mb.items()}
                with self.amp.autocast():
                    logits, value = self.net(mb["obs"])
                dist = Categorical(logits=logits.float())
                value = value.float()
                log_prob = dist.log_prob(mb["actions"])
                log_ratio = log_prob - mb["log_probs"]
                ratio = log_ratio.exp()

                adv = mb["advantages"]
                if adv.numel() > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg_loss = torch.max(
                    -adv * ratio, -adv * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef)
                ).mean()

                v_clipped = mb["values"] + (value - mb["values"]).clamp(
                    -cfg.clip_coef, cfg.clip_coef
                )
                v_loss = (
                    0.5
                    * torch.max(
                        (value - mb["returns"]) ** 2, (v_clipped - mb["returns"]) ** 2
                    ).mean()
                )
                entropy = dist.entropy().mean()
                loss = pg_loss + cfg.value_coef * v_loss - cfg.entropy_coef * entropy
                grad_norm = self.amp.step(
                    loss, self.optimizer, self.net.parameters(), cfg.max_grad_norm
                )

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clip_frac = ((ratio - 1).abs() > cfg.clip_coef).float().mean().item()
                for k, v in zip(
                    stats,
                    (
                        pg_loss.item(),
                        v_loss.item(),
                        entropy.item(),
                        approx_kl,
                        clip_frac,
                        grad_norm,
                    ),
                    strict=True,
                ):
                    stats[k].append(v)
            if cfg.target_kl and stats["approx_kl"][-1] > cfg.target_kl:
                break
        self.net.eval()
        metrics = {k: sum(v) / len(v) for k, v in stats.items()}
        metrics["lr"] = self.optimizer.param_groups[0]["lr"]
        return metrics
