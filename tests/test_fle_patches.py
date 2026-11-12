"""FLE parses every RCON reply with slpp's module-level singleton, whose decode keeps its cursor
on the instance; VecMacroEnv calls it from one thread per env, so the patch serialises it."""

from concurrent.futures import ThreadPoolExecutor

from slpp import slpp as lua

import factorio_agent.env  # noqa: F401  (applies the patch)

TABLE = "{" + ",".join(f'["k{i}"] = {{ ["n"] = {i},["s"] = "v{i}",}}' for i in range(3000)) + "}"


def test_shared_slpp_decodes_correctly_from_many_threads():
    expected = lua.decode(TABLE)
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda _: lua.decode(TABLE), range(32)))
    assert all(r == expected for r in results)
