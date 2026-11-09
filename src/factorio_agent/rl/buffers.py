"""Experience storage: an on-policy rollout buffer (PPO), a replay buffer (MCGS, uniform DQN)
and a prioritized replay buffer (Rainbow DQN)."""

from collections import deque
from collections.abc import Iterator

import torch


class RolloutBuffer:
    """Fixed-length on-policy rollout of `num_steps` x `num_envs` transitions."""

    def __init__(self, num_steps: int, num_envs: int, obs_dim: int):
        shape = (num_steps, num_envs)
        self.obs = torch.zeros(*shape, obs_dim)
        self.actions = torch.zeros(shape, dtype=torch.long)
        self.log_probs = torch.zeros(shape)
        self.rewards = torch.zeros(shape)
        self.dones = torch.zeros(shape)  # 1 where the episode terminated after this step
        self.values = torch.zeros(shape)
        self.advantages = torch.zeros(shape)
        self.returns = torch.zeros(shape)
        self.num_steps = num_steps
        self.t = 0

    def add(self, obs, actions, log_probs, rewards, dones, values) -> None:
        t = self.t
        self.obs[t], self.actions[t], self.log_probs[t] = obs, actions, log_probs
        self.rewards[t], self.dones[t], self.values[t] = rewards, dones, values
        self.t += 1

    def compute_gae(self, last_value: torch.Tensor, gamma: float, gae_lambda: float) -> None:
        """Generalised advantage estimation. `last_value` is V(s_T) for the obs after the
        rollout. Truncations are expected to be bootstrapped into `rewards` already, so
        `dones` only marks true terminations (and episode boundaries)."""
        gae = torch.zeros_like(last_value)
        next_value = last_value
        for t in reversed(range(self.num_steps)):
            not_done = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * not_done - self.values[t]
            gae = delta + gamma * gae_lambda * not_done * gae
            self.advantages[t] = gae
            next_value = self.values[t]
        self.returns = self.advantages + self.values
        self.t = 0

    def minibatches(self, batch_size: int) -> Iterator[dict[str, torch.Tensor]]:
        flat = {
            "obs": self.obs.flatten(0, 1),
            "actions": self.actions.flatten(),
            "log_probs": self.log_probs.flatten(),
            "advantages": self.advantages.flatten(),
            "returns": self.returns.flatten(),
            "values": self.values.flatten(),
        }
        n = flat["actions"].numel()
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            yield {k: v[idx] for k, v in flat.items()}


class ReplayBuffer:
    """Circular buffer of named tensor fields, e.g.
    `ReplayBuffer(10_000, obs=(16,), action=(), reward=())`."""

    def __init__(self, capacity: int, **fields: tuple[int, ...]):
        self.capacity = capacity
        self.data = {
            name: torch.zeros(capacity, *shape, dtype=torch.long if name == "action" else None)
            for name, shape in fields.items()
        }
        self.pos = 0
        self.size = 0

    def add(self, **values: torch.Tensor | float) -> None:
        for name, buf in self.data.items():
            buf[self.pos] = torch.as_tensor(values[name])
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int, device: torch.device | str = "cpu"
    ) -> dict[str, torch.Tensor]:
        idx = torch.randint(0, self.size, (batch_size,))
        return {name: buf[idx].to(device) for name, buf in self.data.items()}

    def __len__(self) -> int:
        return self.size

    def state_dict(self) -> dict[str, object]:
        """Only the filled part, so checkpoints stay small early in training."""
        return {
            "pos": self.pos,
            "size": self.size,
            "data": {k: v[: self.size].clone() for k, v in self.data.items()},
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        size = min(state["size"], self.capacity)
        for k, v in state["data"].items():
            self.data[k][:size] = v[:size]
        self.size = size
        self.pos = state["pos"] % self.capacity


class SumTree:
    """Binary tree of priorities: batched O(log N) updates and proportional lookups.

    Leaves are padded to a power of two so that they all sit at the same depth, which lets a
    batch move up (or down) the tree one level at a time.
    """

    def __init__(self, capacity: int):
        self.leaves = 1 << max(capacity - 1, 0).bit_length()
        self.tree = torch.zeros(2 * self.leaves, dtype=torch.float64)

    @property
    def total(self) -> float:
        return self.tree[1].item()

    def priority(self, idx: torch.Tensor) -> torch.Tensor:
        return self.tree[idx + self.leaves]

    def update(self, idx: torch.Tensor, priority: torch.Tensor) -> None:
        node = idx + self.leaves
        self.tree[node] = priority.to(self.tree.dtype)
        while node[0] > 1:
            # recomputed from the children, so rounding errors cannot accumulate
            node = torch.unique(node // 2)
            self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def find(self, mass: torch.Tensor) -> torch.Tensor:
        """For each value in [0, total), the leaf whose cumulative priority interval holds it."""
        mass = mass.to(self.tree.dtype).clone()
        node = torch.ones_like(mass, dtype=torch.long)
        while node[0] < self.leaves:
            left = 2 * node
            left_mass = self.tree[left]
            # rounding can leave `mass` just past the last positive leaf: never step into an
            # empty subtree, whose transitions do not exist yet
            right = (mass >= left_mass) & (self.tree[left + 1] > 0)
            mass -= left_mass * right
            node = left + right.long()
        return node - self.leaves


class PrioritizedReplayBuffer(ReplayBuffer):
    """Proportional prioritized replay: P(i) = p_i / sum_k p_k, with p_i = (error_i + eps)^omega.

    `sample` adds to the batch the sampled `index` (for `update_priorities`) and the importance
    sampling `weight` (N * P(i))^-beta, scaled by the batch maximum so it only shrinks updates.
    New transitions get the highest priority seen so far, so each is replayed soon after it
    arrives.
    """

    PRIORITY_EPS = 1e-6  # keeps zero-error transitions sampleable

    def __init__(self, capacity: int, omega: float = 0.5, **fields: tuple[int, ...]):
        super().__init__(capacity, **fields)
        self.omega = omega
        self.tree = SumTree(capacity)
        self.max_priority = 1.0

    def add(self, **values: torch.Tensor | float) -> None:
        idx = torch.tensor([self.pos])
        super().add(**values)
        self.tree.update(idx, torch.tensor([self.max_priority]))

    def sample(
        self, batch_size: int, device: torch.device | str = "cpu", beta: float = 1.0
    ) -> dict[str, torch.Tensor]:
        # stratified: one draw from each of batch_size equal-mass segments
        total = self.tree.total
        draws = torch.rand(batch_size, dtype=torch.float64)
        idx = self.tree.find((torch.arange(batch_size) + draws) * total / batch_size)
        weights = (self.size * self.tree.priority(idx) / total) ** -beta
        batch = {name: buf[idx].to(device) for name, buf in self.data.items()}
        batch["index"] = idx
        batch["weight"] = (weights / weights.max()).to(device, torch.float32)
        return batch

    def update_priorities(self, idx: torch.Tensor, errors: torch.Tensor) -> None:
        priority = (errors.detach().double().cpu() + self.PRIORITY_EPS) ** self.omega
        self.max_priority = max(self.max_priority, priority.max().item())
        self.tree.update(idx, priority)

    def state_dict(self) -> dict[str, object]:
        return {
            **super().state_dict(),
            "priorities": self.tree.priority(torch.arange(self.size)).clone(),
            "max_priority": self.max_priority,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        super().load_state_dict(state)
        self.max_priority = state.get("max_priority", 1.0)
        self.tree = SumTree(self.capacity)
        if self.size:
            # a buffer saved without priorities (uniform run): every transition starts equal
            default = torch.full((self.size,), self.max_priority)
            priorities = state.get("priorities", default)[: self.size]
            self.tree.update(torch.arange(self.size), priorities)


class NStepAccumulator:
    """Turns one env's 1-step transitions into n-step ones:
    (s_t, a_t, sum_k gamma^k r_{t+k}, s_{t+n}, done, gamma^n).

    Call `push` every step; it returns the transitions that are ready. When an episode ends
    the remaining partial windows are flushed. `done` marks a true termination (no bootstrap);
    on truncation pass `terminated=False` with the real successor obs.
    """

    def __init__(self, n: int, gamma: float):
        self.n = n
        self.gamma = gamma
        self.window: deque[tuple[torch.Tensor, int, float]] = deque()

    def push(
        self,
        obs: torch.Tensor,
        action: int,
        reward: float,
        next_obs: torch.Tensor,
        terminated: bool,
        episode_end: bool,
    ) -> list[dict[str, torch.Tensor | float]]:
        self.window.append((obs, action, reward))
        out = []
        if len(self.window) == self.n:
            out.append(self._emit(next_obs, terminated))
            self.window.popleft()
        if episode_end:
            while self.window:
                out.append(self._emit(next_obs, terminated))
                self.window.popleft()
        return out

    def _emit(self, next_obs: torch.Tensor, terminated: bool) -> dict[str, torch.Tensor | float]:
        ret = sum(r * self.gamma**k for k, (_, _, r) in enumerate(self.window))
        obs, action, _ = self.window[0]
        return {
            "obs": obs,
            "action": action,
            "reward": ret,
            "next_obs": next_obs,
            "done": float(terminated),
            "discount": self.gamma ** len(self.window),
        }
