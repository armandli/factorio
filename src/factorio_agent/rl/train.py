"""Train an RL agent on FLE macro-actions.

uv run factorio-train ppo --mock --total-steps 2000      # offline, no server needed
uv run factorio-train dqn --num-envs 2 --total-steps 500  # needs `fle cluster start -n 2`
uv run factorio-train mcgs --num-simulations 64
uv run factorio-train ppo --resume runs/ppo-.../checkpoint-final.pt --total-steps 4000
uv run factorio-train ppo --help                          # all hyperparameters
"""

import argparse
import dataclasses
import signal
import sys
from importlib import import_module

from factorio_agent.rl import ALGORITHMS, load_trainer, make_vec_env
from factorio_agent.rl.common import TrainConfig


def add_config_args(parser: argparse.ArgumentParser, cls: type[TrainConfig]) -> None:
    for f in dataclasses.fields(cls):
        flag = "--" + f.name.replace("_", "-")
        if f.type is bool or f.type == "bool":
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default)
        else:
            parser.add_argument(flag, type=type(f.default), default=f.default)


def _interrupt(signum, frame):
    # ignore repeats (Ctrl-C reaches every process in the group) so the checkpoint
    # saved on the way out isn't interrupted
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise KeyboardInterrupt(f"signal {signum}")


def main(argv: list[str] | None = None) -> int:
    # SIGINT/SIGTERM: the trainer saves a checkpoint, then exits
    previous = {sig: signal.signal(sig, _interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        return _main(argv)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _main(argv: list[str] | None) -> int:
    parser = argparse.ArgumentParser(
        prog="factorio-train",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="algo", required=True)
    for name, trainer_cls in ALGORITHMS.items():
        summary = import_module(trainer_cls.__module__).__doc__.splitlines()[0]
        p = sub.add_parser(
            name,
            help=summary,
            description=summary,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        add_config_args(p, trainer_cls.config_cls)
        p.add_argument("--resume", default="", help="checkpoint to continue training from")
    args = vars(parser.parse_args(argv))
    algo, resume = args.pop("algo"), args.pop("resume")
    trainer_cls = ALGORITHMS[algo]
    cfg = trainer_cls.config_cls(**args)

    try:
        envs = make_vec_env(
            cfg.env_id,
            cfg.num_envs,
            cfg.max_episode_steps,
            cfg.error_penalty,
            mock=cfg.mock,
            seed=cfg.seed,
            first_container=cfg.first_container,
        )
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    try:
        if resume:
            trainer = load_trainer(resume, envs, **dataclasses.asdict(cfg))
        else:
            trainer = trainer_cls(cfg, envs)
        print(
            f"{algo} on {'mock env' if cfg.mock else cfg.env_id}, {cfg.num_envs} env(s), "
            f"device={trainer.device}"
        )
        trainer.train()
    finally:
        envs.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
