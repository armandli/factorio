"""Agents that map FLE observations to Python-code actions."""

import random
from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn

from factorio_agent.actions import MACRO_ACTIONS, NUM_ACTIONS, MacroAction
from factorio_agent.features import FEATURE_DIM, featurize


class Agent(ABC):
    """Chooses a MacroAction given the latest observation."""

    @abstractmethod
    def act(self, obs: dict[str, Any]) -> MacroAction: ...

    def reset(self) -> None:
        """Called at the start of each episode."""


class RandomAgent(Agent):
    """Uniformly random macro-actions."""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def act(self, obs: dict[str, Any]) -> MacroAction:
        return self.rng.choice(MACRO_ACTIONS)


class ScriptedAgent(Agent):
    """Cycles through a fixed plan: look around, gather fuel and ore, place a fuelled drill."""

    PLAN = (
        "inspect_inventory",
        "harvest_coal",
        "harvest_iron_ore",
        "place_iron_drill",
        "wait",
        "list_entities",
        "harvest_stone",
        "inspect_inventory",
    )

    def __init__(self):
        self._by_name = {a.name: a for a in MACRO_ACTIONS}
        self._t = 0

    def reset(self) -> None:
        self._t = 0

    def act(self, obs: dict[str, Any]) -> MacroAction:
        action = self._by_name[self.PLAN[self._t % len(self.PLAN)]]
        self._t += 1
        return action


class PolicyNet(nn.Module):
    """Small MLP actor-critic: features -> (action logits, state value)."""

    def __init__(
        self, obs_dim: int = FEATURE_DIM, num_actions: int = NUM_ACTIONS, hidden: int = 64
    ):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, num_actions)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)


class PolicyAgent(Agent):
    """Samples macro-actions from a PyTorch policy network.

    Weights are random unless `net` comes from training (`factorio-train ppo`).
    """

    def __init__(self, net: PolicyNet | None = None, seed: int | None = None, device: str = "cpu"):
        self.generator = torch.Generator(device=device)
        if seed is not None:
            torch.manual_seed(seed)
            self.generator.manual_seed(seed)
        self.device = torch.device(device)
        self.net = (net or PolicyNet()).to(self.device).eval()
        self.last_log_prob: torch.Tensor | None = None
        self.last_value: torch.Tensor | None = None

    @torch.no_grad()
    def act(self, obs: dict[str, Any]) -> MacroAction:
        x = featurize(obs).to(self.device)
        logits, value = self.net(x)
        probs = torch.softmax(logits, dim=-1)
        idx = torch.multinomial(probs, 1, generator=self.generator).item()
        self.last_log_prob = torch.log(probs[idx])
        self.last_value = value
        return MACRO_ACTIONS[idx]


AGENTS: dict[str, type[Agent]] = {
    "random": RandomAgent,
    "scripted": ScriptedAgent,
    "policy": PolicyAgent,
}
