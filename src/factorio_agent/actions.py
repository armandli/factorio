"""Discrete macro-action library.

FLE actions are Python programs executed in the game's REPL namespace, where tools such as
`nearest`, `move_to`, `harvest_resource`, `place_entity`, `Prototype` and `Resource` are in
scope. A learned policy picks an index into MACRO_ACTIONS, giving a standard discrete action
space on top of FLE's free-form code actions.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MacroAction:
    name: str
    code: str


def _harvest(resource: str, quantity: int = 10) -> str:
    return (
        f"pos = nearest(Resource.{resource})\n"
        f"move_to(pos)\n"
        f"n = harvest_resource(pos, quantity={quantity})\n"
        f"print(f'harvested {{n}} {resource}')"
    )


MACRO_ACTIONS: tuple[MacroAction, ...] = (
    MacroAction("inspect_inventory", "print(inspect_inventory())"),
    MacroAction("harvest_iron_ore", _harvest("IronOre")),
    MacroAction("harvest_copper_ore", _harvest("CopperOre")),
    MacroAction("harvest_coal", _harvest("Coal")),
    MacroAction("harvest_stone", _harvest("Stone")),
    MacroAction(
        "place_iron_drill",
        "pos = nearest(Resource.IronOre)\n"
        "move_to(pos)\n"
        "drill = place_entity(Prototype.BurnerMiningDrill, position=pos)\n"
        "drill = insert_item(Prototype.Coal, drill, quantity=5)\n"
        "print(drill)",
    ),
    MacroAction(
        "craft_gear_wheel",
        "n = craft_item(Prototype.IronGearWheel, quantity=1)\nprint(f'crafted {n} iron-gear-wheel')",
    ),
    MacroAction("list_entities", "print(get_entities())"),
    MacroAction("wait", "sleep(5)\nprint('waited 5s')"),
)

NUM_ACTIONS = len(MACRO_ACTIONS)
