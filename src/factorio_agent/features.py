"""Convert FLE dict observations into fixed-size tensors for PyTorch policies."""

import math
from typing import Any

import torch

# Inventory items tracked as features; everything else is ignored.
FEATURE_ITEMS: tuple[str, ...] = (
    "iron-ore",
    "copper-ore",
    "coal",
    "stone",
    "wood",
    "iron-plate",
    "copper-plate",
    "iron-gear-wheel",
    "burner-mining-drill",
    "stone-furnace",
    "wooden-chest",
    "transport-belt",
)

# inventory counts + entity count + score + game time + last-action-error flag
FEATURE_DIM = len(FEATURE_ITEMS) + 4


def featurize(obs: dict[str, Any]) -> torch.Tensor:
    """Map an observation dict (as returned by FactorioGymEnv) to a float tensor [FEATURE_DIM].

    Counts are log1p-scaled so values from 0 to thousands land in a similar range.
    """
    inventory = {item["type"]: int(item["quantity"]) for item in obs.get("inventory", [])}
    counts = [inventory.get(name, 0) for name in FEATURE_ITEMS]

    raw_text = str(obs.get("raw_text", "")).lower()
    errored = float("error" in raw_text or "exception" in raw_text)
    game_time = float(obs.get("game_info", {}).get("time", 0.0))

    values = [math.log1p(c) for c in counts] + [
        math.log1p(len(obs.get("entities", []))),
        math.copysign(math.log1p(abs(float(obs.get("score", 0.0)))), float(obs.get("score", 0.0))),
        math.log1p(game_time),
        errored,
    ]
    return torch.tensor(values, dtype=torch.float32)
