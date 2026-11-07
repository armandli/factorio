"""A tiny offline stand-in for FLE, for developing and smoke-testing training code.

It speaks FLE's dict API and understands the MACRO_ACTIONS programs, with toy dynamics:
harvesting adds 8-12 of a resource, `place_iron_drill` turns a drill plus 5 coal into a
placed drill, and every step each placed drill produces 1 iron ore, which is the reward
(FLE rewards production score). The best policy places all drills early, fetching coal when
it runs out. Nothing here is faithful to real Factorio beyond that shape.
"""

import random
from dataclasses import dataclass
from typing import Any

from factorio_agent.actions import MACRO_ACTIONS


@dataclass
class CodeAction:
    code: str
    agent_idx: int = 0


START_INVENTORY = {"burner-mining-drill": 5, "coal": 10, "stone-furnace": 2, "wooden-chest": 2}
HARVEST = {
    "harvest_iron_ore": "iron-ore",
    "harvest_copper_ore": "copper-ore",
    "harvest_coal": "coal",
    "harvest_stone": "stone",
}
TICKS_PER_STEP = 300


class MockFactorioEnv:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self._by_code = {a.code: a.name for a in MACRO_ACTIONS}
        self.inventory: dict[str, int] = {}
        self.drills = 0
        self.ticks = 0
        self.score = 0.0

    def reset(self, options: dict[str, Any] | None = None, seed: int | None = None):
        self.inventory = dict(START_INVENTORY)
        self.drills = 0
        self.ticks = 0
        self.score = 0.0
        return self._obs(""), {}

    def step(self, action: Any):
        name = self._by_code.get(action.code)
        text, error = self._apply(name)
        reward = float(self.drills)
        self.inventory["iron-ore"] = self.inventory.get("iron-ore", 0) + self.drills
        self.score += reward
        self.ticks += TICKS_PER_STEP
        info = {"error_occurred": error, "ticks": self.ticks}
        return self._obs(text), reward, False, False, info

    def close(self) -> None:
        pass

    def _apply(self, name: str | None) -> tuple[str, bool]:
        inv = self.inventory
        if name in HARVEST:
            item = HARVEST[name]
            n = self.rng.randint(8, 12)
            inv[item] = inv.get(item, 0) + n
            return f"harvested {n} {item}", False
        if name == "place_iron_drill":
            if inv.get("burner-mining-drill", 0) < 1 or inv.get("coal", 0) < 5:
                return "Error: missing burner-mining-drill or coal", True
            inv["burner-mining-drill"] -= 1
            inv["coal"] -= 5
            self.drills += 1
            return "placed BurnerMiningDrill", False
        if name == "craft_gear_wheel":
            if inv.get("iron-plate", 0) < 2:
                return "Exception: requires 2 iron-plate", True
            inv["iron-plate"] -= 2
            inv["iron-gear-wheel"] = inv.get("iron-gear-wheel", 0) + 1
            return "crafted 1 iron-gear-wheel", False
        if name is None:
            return "Error: unknown program", True
        return "ok", False

    def _obs(self, text: str) -> dict[str, Any]:
        return {
            "raw_text": text,
            "inventory": [{"type": k, "quantity": v} for k, v in self.inventory.items() if v],
            "entities": [f"BurnerMiningDrill({i})" for i in range(self.drills)],
            "score": self.score,
            "game_info": {"tick": self.ticks, "time": self.ticks / 60, "speed": 1.0},
        }
