"""Monte-Carlo Graph Search (MCGS) with PUCT selection over a pluggable model.

Plain MCTS builds a tree, so the same game state reached by different action orders
("harvest coal, then ore" vs "ore, then coal") gets separate, independently-estimated
nodes. MCGS keys nodes by state in a transposition table, turning the tree into a DAG:

- a node's value is the mean of every simulation through it, whichever path it came by;
- an edge's Q is computed as r + gamma * V(child), so it sees the child's shared value;
- a simulation that would revisit a node already on its own path stops there (no cycles).

Reference: Czech, Korus, Kersting, "Improving AlphaZero Using Monte-Carlo Graph Search"
(ICAPS 2021). This is the simple variant without their Q-delta correction.

The model is anything implementing `SearchModel`: `LearnedModel` (prediction net + learned
dynamics in feature space) is used for training, but a simulator-backed model (e.g. FLE
game-state snapshots) would also fit.
"""

import math
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Protocol

import torch

from factorio_agent.rl.networks import PolicyNet, WorldModel


class SearchModel(Protocol):
    def evaluate(self, state: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Prior action probabilities [A] and value estimate for `state`."""
        ...

    def step(self, state: torch.Tensor, action: int) -> tuple[torch.Tensor, float, bool]:
        """(next_state, reward, terminal)."""
        ...


class LearnedModel:
    def __init__(self, prediction: PolicyNet, dynamics: WorldModel, device: torch.device):
        self.prediction = prediction
        self.dynamics = dynamics
        self.device = device

    @torch.no_grad()
    def evaluate(self, state: torch.Tensor) -> tuple[torch.Tensor, float]:
        logits, value = self.prediction(state.to(self.device))
        return torch.softmax(logits.float(), -1).cpu(), float(value)

    @torch.no_grad()
    def step(self, state: torch.Tensor, action: int) -> tuple[torch.Tensor, float, bool]:
        s = state.to(self.device).unsqueeze(0)
        a = torch.tensor([action], device=self.device)
        next_state, reward, done_logit = self.dynamics(s, a)
        return next_state[0].float().cpu(), float(reward), bool(done_logit > 0)


@dataclass
class Edge:
    child: "Node"
    reward: float
    visits: int = 0


@dataclass
class Node:
    state: torch.Tensor
    prior: torch.Tensor
    net_value: float
    terminal: bool
    visits: int = 0
    value_sum: float = 0.0
    edges: dict[int, Edge] = field(default_factory=dict)

    @property
    def value(self) -> float:
        if self.terminal:
            return 0.0
        return self.value_sum / self.visits if self.visits else self.net_value


class MinMaxStats:
    """Normalises Q to [0, 1] over values seen in this search (rewards are unbounded)."""

    def __init__(self):
        self.lo, self.hi = math.inf, -math.inf

    def update(self, v: float) -> None:
        self.lo, self.hi = min(self.lo, v), max(self.hi, v)

    def normalize(self, v: float) -> float:
        return (v - self.lo) / (self.hi - self.lo) if self.hi > self.lo else 0.5


@dataclass
class SearchResult:
    policy: torch.Tensor  # root visit distribution [A]
    value: float  # root value estimate
    num_nodes: int  # distinct states in the graph (< simulations when transpositions merge)


class MCGS:
    def __init__(
        self,
        model: SearchModel,
        num_actions: int,
        num_simulations: int = 32,
        gamma: float = 0.99,
        c_puct: float = 1.25,
        key_decimals: int = 1,
        dirichlet_alpha: float = 0.3,
        dirichlet_frac: float = 0.25,
    ):
        self.model = model
        self.num_actions = num_actions
        self.num_simulations = num_simulations
        self.gamma = gamma
        self.c_puct = c_puct
        self.key_scale = 10.0**key_decimals
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_frac = dirichlet_frac

    def key(self, state: torch.Tensor) -> Hashable:
        """States that round to the same grid point are treated as the same node."""
        return tuple(torch.round(state * self.key_scale).long().tolist())

    def search(self, root_state: torch.Tensor, add_noise: bool = False) -> SearchResult:
        table: dict[Hashable, Node] = {}
        root = self._new_node(root_state, terminal=False, table=table)
        if add_noise and self.dirichlet_frac > 0:
            noise = torch.distributions.Dirichlet(
                torch.full((self.num_actions,), self.dirichlet_alpha)
            ).sample()
            root.prior = (1 - self.dirichlet_frac) * root.prior + self.dirichlet_frac * noise
        stats = MinMaxStats()

        for _ in range(self.num_simulations):
            node, path, on_path = root, [], {id(root)}
            while True:
                if node.terminal:
                    leaf_value = 0.0
                    break
                action = self._select(node, stats)
                edge = node.edges.get(action)
                if edge is None:  # expand: one model step, then evaluate the leaf
                    edge = self._expand(node, action, table)
                    path.append((node, edge))
                    leaf_value = edge.child.value
                    node = edge.child
                    break
                path.append((node, edge))
                node = edge.child
                if id(node) in on_path:  # cycle: stop and use the node's current estimate
                    leaf_value = node.value
                    break
                on_path.add(id(node))
            self._backup(path, node, leaf_value, stats)

        visits = torch.tensor(
            [root.edges[a].visits if a in root.edges else 0 for a in range(self.num_actions)],
            dtype=torch.float32,
        )
        return SearchResult(visits / visits.sum(), root.value, len(table))

    def _new_node(self, state: torch.Tensor, terminal: bool, table: dict) -> Node:
        prior, value = self.model.evaluate(state)
        node = Node(state=state, prior=prior, net_value=value, terminal=terminal)
        table[self.key(state)] = node
        return node

    def _expand(self, node: Node, action: int, table: dict) -> Edge:
        next_state, reward, terminal = self.model.step(node.state, action)
        child = table.get(self.key(next_state))  # transposition: reuse the existing node
        if child is None:
            child = self._new_node(next_state, terminal, table)
        edge = Edge(child=child, reward=reward)
        node.edges[action] = edge
        return edge

    def _q(self, edge: Edge) -> float:
        return edge.reward + self.gamma * edge.child.value

    def _select(self, node: Node, stats: MinMaxStats) -> int:
        sqrt_n = math.sqrt(max(node.visits, 1))
        # first-play urgency: unexpanded actions are scored at the parent's value
        fpu = stats.normalize(node.value)
        best, best_score = 0, -math.inf
        for a in range(self.num_actions):
            edge = node.edges.get(a)
            q = stats.normalize(self._q(edge)) if edge else fpu
            n = edge.visits if edge else 0
            score = q + self.c_puct * float(node.prior[a]) * sqrt_n / (1 + n)
            if score > best_score:
                best, best_score = a, score
        return best

    def _backup(
        self, path: list[tuple[Node, Edge]], leaf: Node, value: float, stats: MinMaxStats
    ) -> None:
        leaf.visits += 1
        leaf.value_sum += value
        g = value
        for node, edge in reversed(path):
            g = edge.reward + self.gamma * g
            edge.visits += 1
            node.visits += 1
            node.value_sum += g
            stats.update(self._q(edge))
