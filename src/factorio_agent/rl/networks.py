"""Networks used by the RL algorithms. All take featurized observations [B, FEATURE_DIM].

- PPO: `PolicyNet` (actor-critic, from factorio_agent.agents).
- DQN: `QNetwork` (dueling Q-values).
- MCGS: `PolicyNet` as the prediction net (prior + value) and `WorldModel` as the learned
  dynamics model searched over.
"""

import torch
import torch.nn.functional as F
from torch import nn

from factorio_agent.actions import NUM_ACTIONS
from factorio_agent.agents import PolicyNet
from factorio_agent.features import FEATURE_DIM

__all__ = ["PolicyNet", "QNetwork", "WorldModel", "mlp"]


def mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    mods: list[nn.Module] = []
    d = in_dim
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


class QNetwork(nn.Module):
    """Dueling head: Q(s, a) = V(s) + A(s, a) - mean_a A(s, a)."""

    def __init__(
        self, obs_dim: int = FEATURE_DIM, num_actions: int = NUM_ACTIONS, hidden: int = 128
    ):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.value = nn.Linear(hidden, 1)
        self.advantage = nn.Linear(hidden, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        a = self.advantage(h)
        return self.value(h) + a - a.mean(dim=-1, keepdim=True)


class WorldModel(nn.Module):
    """Learned dynamics in feature space: (s, a) -> (s', r, P(terminal)).

    Predicts s' as a residual on s, so "nothing changed" is the easy default. Working in
    observation space (not a latent) keeps states comparable, which lets MCGS merge
    transpositions such as "harvest coal, then ore" vs "harvest ore, then coal".
    """

    def __init__(
        self, obs_dim: int = FEATURE_DIM, num_actions: int = NUM_ACTIONS, hidden: int = 128
    ):
        super().__init__()
        self.num_actions = num_actions
        self.body = nn.Sequential(
            nn.Linear(obs_dim + num_actions, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.delta_head = nn.Linear(hidden, obs_dim)
        self.reward_head = nn.Linear(hidden, 1)
        self.done_head = nn.Linear(hidden, 1)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (next_obs, reward, done_logit)."""
        a = F.one_hot(action.long(), self.num_actions).to(obs.dtype)
        h = self.body(torch.cat([obs, a], dim=-1))
        next_obs = obs + self.delta_head(h)
        return next_obs, self.reward_head(h).squeeze(-1), self.done_head(h).squeeze(-1)
