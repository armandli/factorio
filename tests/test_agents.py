"""Offline tests: no Factorio server needed."""

import ast

import torch

from factorio_agent import MACRO_ACTIONS, PolicyAgent, RandomAgent, ScriptedAgent, featurize
from factorio_agent.agents import PolicyNet
from factorio_agent.features import FEATURE_DIM

OBS = {
    "raw_text": "",
    "inventory": [
        {"type": "coal", "quantity": 500},
        {"type": "burner-mining-drill", "quantity": 50},
    ],
    "entities": [],
    "score": 0.0,
    "game_info": {"tick": 0, "time": 0.0, "speed": 1.0},
}


def test_macro_actions_are_valid_python():
    for action in MACRO_ACTIONS:
        ast.parse(action.code)
    assert len({a.name for a in MACRO_ACTIONS}) == len(MACRO_ACTIONS)


def test_featurize_shape_and_values():
    x = featurize(OBS)
    assert x.shape == (FEATURE_DIM,)
    assert x.dtype == torch.float32
    assert torch.isfinite(x).all()
    assert x.sum() > 0


def test_featurize_flags_errors():
    x = featurize({**OBS, "raw_text": "Error: could not place entity"})
    assert x[-1] == 1.0


def test_policy_net_forward_batch():
    logits, value = PolicyNet()(torch.randn(4, FEATURE_DIM))
    assert logits.shape == (4, len(MACRO_ACTIONS))
    assert value.shape == (4,)


def test_agents_return_macro_actions():
    for agent in (RandomAgent(seed=0), ScriptedAgent(), PolicyAgent(seed=0)):
        agent.reset()
        for _ in range(3):
            assert agent.act(OBS) in MACRO_ACTIONS


def test_policy_agent_seeded_is_deterministic():
    a = [PolicyAgent(seed=1).act(OBS).name for _ in range(2)]
    assert a[0] == a[1]
