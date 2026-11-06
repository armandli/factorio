"""Run an untrained agent in a Factorio Learning Environment task.

uv run factorio-agent --list
uv run factorio-agent --env iron_ore_throughput --agent policy --steps 10
"""

import argparse
import sys
import textwrap

from factorio_agent.agents import AGENTS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="factorio-agent",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--env", default="iron_ore_throughput", help="FLE task / env id")
    parser.add_argument("--agent", choices=sorted(AGENTS), default="policy")
    parser.add_argument("--steps", type=int, default=10, help="max steps in the episode")
    parser.add_argument("--run-idx", type=int, default=0, help="which Factorio container to use")
    parser.add_argument("--list", action="store_true", help="list available env ids and exit")
    args = parser.parse_args(argv)

    from factorio_agent import env as fenv

    if args.list:
        for env_id in fenv.list_available_environments():
            print(f"{env_id:60s} {fenv.get_environment_info(env_id)['description'][:80]}")
        return 0

    try:
        containers = fenv.check_cluster()
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"Factorio containers: {', '.join(containers)}")

    env = fenv.make_env(args.env, run_idx=args.run_idx)
    agent = AGENTS[args.agent]()

    def show(r: fenv.StepRecord) -> None:
        status = "ERROR" if r.error else "ok"
        print(f"\n[step {r.step}] {r.action}  reward={r.reward:.3f}  ticks={r.ticks}  {status}")
        print(textwrap.indent(textwrap.shorten(r.output, 600, placeholder=" ..."), "  "))

    try:
        records = fenv.run_episode(env, agent, max_steps=args.steps, on_step=show)
    finally:
        env.close()

    total = sum(r.reward for r in records)
    errors = sum(r.error for r in records)
    print(f"\nEpisode done: {len(records)} steps, total reward={total:.3f}, errors={errors}")
    print(f"Final inventory: {records[-1].inventory if records else {}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
