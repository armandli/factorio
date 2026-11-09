"""Networks used by the RL algorithms. All take featurized observations [B, FEATURE_DIM].

- PPO: `PolicyNet` (actor-critic, from factorio_agent.agents).
- DQN: `QNetwork` (Rainbow: categorical returns, dueling streams, noisy layers).
- MCGS: `PolicyNet` as the prediction net (prior + value) and `WorldModel` as the learned
  dynamics model searched over.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from factorio_agent.actions import NUM_ACTIONS
from factorio_agent.agents import PolicyNet
from factorio_agent.features import FEATURE_DIM

__all__ = ["NoisyLinear", "PolicyNet", "QNetwork", "WorldModel", "categorical_projection", "mlp"]


def mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    mods: list[nn.Module] = []
    d = in_dim
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


class NoisyLinear(nn.Module):
    """Linear layer with learned factorised Gaussian noise (Fortunato et al.):
    y = (b + W x) + (b_sigma * e_b + (W_sigma * e_w) x).

    Noise applies in train mode, until the next `reset_noise`; eval mode uses the mean weights.
    """

    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5):
        super().__init__()
        bound = 1.0 / math.sqrt(in_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(out_features).uniform_(-bound, bound))
        self.weight_sigma = nn.Parameter(torch.full((out_features, in_features), sigma0 * bound))
        self.bias_sigma = nn.Parameter(torch.full((out_features,), sigma0 * bound))
        # redrawn constantly, so there is no reason to checkpoint them
        self.register_buffer("eps_in", torch.zeros(in_features), persistent=False)
        self.register_buffer("eps_out", torch.zeros(out_features), persistent=False)
        self.reset_noise()

    @torch.no_grad()
    def reset_noise(self) -> None:
        for eps in (self.eps_in, self.eps_out):
            eps.normal_()
            eps.copy_(eps.sign() * eps.abs().sqrt())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return F.linear(x, self.weight, self.bias)
        weight = self.weight + self.weight_sigma * torch.outer(self.eps_out, self.eps_in)
        return F.linear(x, weight, self.bias + self.bias_sigma * self.eps_out)


class QNetwork(nn.Module):
    """Rainbow's network: a shared encoder feeding value and advantage streams, aggregated per
    atom into the logits of a categorical return distribution for each action:

        logit_i(s, a) = V_i(s) + A_i(s, a) - mean_a' A_i(s, a')
        Q(s, a) = sum_i z_i softmax(logit(s, a))_i

    The stream layers are noisy; the encoder stays deterministic, like the paper's conv layers.
    Each component can be switched off: `distributional=False` gives one scalar Q per action,
    `dueling=False` drops the value stream and `noisy=False` uses plain linear layers.
    """

    support: torch.Tensor | None

    def __init__(
        self,
        obs_dim: int = FEATURE_DIM,
        num_actions: int = NUM_ACTIONS,
        hidden: int = 128,
        distributional: bool = True,
        atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        dueling: bool = True,
        noisy: bool = True,
        sigma0: float = 0.5,
    ):
        super().__init__()
        if distributional and (atoms < 2 or v_max <= v_min):
            raise ValueError(f"need atoms >= 2 and v_max > v_min, got {atoms}, {v_min}, {v_max}")
        self.num_actions = num_actions
        self.atoms = atoms if distributional else 1

        def stream(out_dim: int) -> nn.Sequential:
            if noisy:
                layers = NoisyLinear(hidden, hidden, sigma0), NoisyLinear(hidden, out_dim, sigma0)
            else:
                layers = nn.Linear(hidden, hidden), nn.Linear(hidden, out_dim)
            return nn.Sequential(layers[0], nn.ReLU(), layers[1])

        self.encoder = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU())
        self.advantage = stream(num_actions * self.atoms)
        self.value = stream(self.atoms) if dueling else None
        support = torch.linspace(v_min, v_max, atoms) if distributional else None
        self.register_buffer("support", support)

    def reset_noise(self) -> None:
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """[..., num_actions, atoms]; atoms is 1 for a scalar network, whose logits are the Qs."""
        h = self.encoder(x)
        a = self.advantage(h).view(*h.shape[:-1], self.num_actions, self.atoms)
        if self.value is None:
            return a
        v = self.value(h).view(*h.shape[:-1], 1, self.atoms)
        return v + a - a.mean(dim=-2, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Q-values, [..., num_actions]."""
        logits = self.logits(x)
        if self.support is None:
            return logits.squeeze(-1)
        return (logits.softmax(dim=-1) * self.support).sum(dim=-1)


def categorical_projection(
    next_probs: torch.Tensor, returns: torch.Tensor, discounts: torch.Tensor, support: torch.Tensor
) -> torch.Tensor:
    """Project (returns + discounts * z, next_probs) onto the fixed support z.

    Each shifted atom splits its mass linearly between its two neighbours on the grid, so the
    mean of the distribution is preserved unless atoms are clipped at the ends of the support.
    """
    atoms = support.numel()
    v_min, v_max = support[0], support[-1]
    tz = (returns.unsqueeze(-1) + discounts.unsqueeze(-1) * support).clamp(v_min, v_max)
    b = ((tz - v_min) / (v_max - v_min) * (atoms - 1)).clamp(0, atoms - 1)
    lower = b.floor().long()
    upper = (lower + 1).clamp_max(atoms - 1)
    # a point exactly on the grid has upper_w = 0, so all its mass stays on `lower`, and
    # clamping `upper` keeps the top atom in range without losing mass
    upper_w = b - lower.to(b.dtype)
    target = torch.zeros_like(next_probs)
    target.scatter_add_(-1, lower, next_probs * (1.0 - upper_w))
    target.scatter_add_(-1, upper, next_probs * upper_w)
    return target


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
