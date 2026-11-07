"""Evaluation: play a policy without exploration, compare it with baselines, keep the best.

Training returns come from the *exploring* policy (PPO samples, DQN is epsilon-greedy, MCGS
adds Dirichlet noise and samples by temperature), so they are noisy and lag behind what the
agent can actually do. With `--eval-every N` the trainer snapshots its weights every N env
steps and an `Evaluator` plays `--eval-episodes` episodes with them on its own env (a separate
Factorio container) in a background thread, so training keeps going:

- PPO samples from its policy, DQN acts greedily, MCGS plays the most-visited action of a
  noise-free search (each algorithm's `make_agent`);
- the random and scripted agents are evaluated once per run on the same env, as fixed
  reference points (`baselines.json`, reused when a run is resumed);
- the snapshot with the best mean eval return is saved as `checkpoint-best.pt` (`best.json`
  records its step and score).

Rewards include the training error penalty, so eval and training returns are comparable.

uv run factorio-eval runs/ppo-.../checkpoint-best.pt runs/dqn-.../checkpoint-best.pt
uv run factorio-eval runs/ppo-.../checkpoint-final.pt --mock --episodes 10
"""

import argparse
import json
import math
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from factorio_agent.actions import MACRO_ACTIONS
from factorio_agent.agents import Agent, RandomAgent, ScriptedAgent
from factorio_agent.rl.envs import MacroEnv

ACTION_INDEX = {a.name: i for i, a in enumerate(MACRO_ACTIONS)}
BASELINES = ("random", "scripted")


@dataclass
class EvalResult:
    name: str  # "policy" or a baseline name
    step: int  # training step of the evaluated weights (0 for baselines)
    returns: list[float] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    errors: list[int] = field(default_factory=list)

    @property
    def mean_return(self) -> float:
        return sum(self.returns) / len(self.returns)

    @property
    def std_return(self) -> float:
        m = self.mean_return
        return math.sqrt(sum((r - m) ** 2 for r in self.returns) / len(self.returns))

    @property
    def mean_length(self) -> float:
        return sum(self.lengths) / len(self.lengths)

    @property
    def mean_errors(self) -> float:
        return sum(self.errors) / len(self.errors)

    def summary(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "return": self.mean_return,
            "return_std": self.std_return,
            "length": self.mean_length,
            "errors": self.mean_errors,
            "episodes": len(self.returns),
            "returns": self.returns,
        }


def evaluate(
    agent: Agent, env: MacroEnv, episodes: int, name: str = "policy", step: int = 0
) -> EvalResult:
    """Play `episodes` full episodes (up to the env's step limit) and record their returns."""
    result = EvalResult(name, step)
    for _ in range(episodes):
        env.reset()
        agent.reset()
        ret, length, errors = 0.0, 0, 0
        while True:
            macro = agent.act(env.raw_obs)
            s = env.step(ACTION_INDEX[macro.name])
            ret, length, errors = ret + s.reward, length + 1, errors + s.error
            if s.terminated or s.truncated:
                break
        result.returns.append(ret)
        result.lengths.append(length)
        result.errors.append(errors)
    return result


def baseline_agent(name: str, seed: int = 0) -> Agent:
    return RandomAgent(seed=seed) if name == "random" else ScriptedAgent()


def _to_cpu(obj: Any) -> Any:
    """Deep copy of a (nested) state dict with every tensor cloned to the CPU."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj


class Evaluator:
    """Runs evaluations one at a time in a background thread and tracks the best snapshot.

    `submit(ckpt)` queues a checkpoint dict (as built by `Trainer.checkpoint_dict`); if an
    evaluation is already running, it waits, and a newer submission replaces a waiting one.
    `poll()` (call it from the training loop) returns finished results, saves
    `checkpoint-best.pt` on a new best and starts the next queued job. An exception in the
    worker (e.g. the eval server stopped responding) is re-raised from `poll()`.
    """

    def __init__(
        self,
        env: MacroEnv,
        run: Path,
        episodes: int,
        agent_from_checkpoint: Callable[[dict[str, Any]], Agent],
        baselines: bool = True,
        baseline_key: dict[str, Any] | None = None,
        seed: int = 0,
    ):
        self.env = env
        self.run = Path(run)
        self.episodes = episodes
        self.agent_from_checkpoint = agent_from_checkpoint
        self.seed = seed
        self.baseline_key = baseline_key or {}
        self.baselines: dict[str, dict[str, Any]] = {}
        self.best: dict[str, Any] | None = None
        best_path = self.run / "best.json"
        if best_path.exists():  # resumed run: keep competing against the earlier best
            self.best = json.loads(best_path.read_text())
        self._thread: threading.Thread | None = None
        self._job_out: list[Any] = []
        self._pending: dict[str, Any] | None = None
        self._last_submitted_step: int | None = None
        if baselines:
            cached = self._cached_baselines()
            if cached:
                self.baselines = cached
            else:
                self._launch(self._baseline_job)

    # -- public API -------------------------------------------------------------------------

    def submit(self, ckpt: dict[str, Any]) -> None:
        self._pending = _to_cpu(ckpt)
        self._last_submitted_step = ckpt["global_step"]
        if self._thread is None:  # a finished but unpolled job keeps its slot until poll()
            self._launch_pending()

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def poll(self) -> list[EvalResult]:
        """Results of a finished job (empty if none finished), then start the queued job."""
        if self._thread is None or self._thread.is_alive():
            return []
        self._thread.join()
        self._thread = None
        out, self._job_out = self._job_out, []
        if out and isinstance(out[0], BaseException):
            raise RuntimeError(f"evaluation failed: {out[0]!r}") from out[0]
        results: list[EvalResult] = []
        for res, ckpt in out:
            results.append(res)
            if res.name in BASELINES:
                self.baselines[res.name] = res.summary()
            else:
                self._update_best(res, ckpt)
        if any(r.name in BASELINES for r in results):
            (self.run / "baselines.json").write_text(
                json.dumps({"key": self.baseline_key, **self.baselines}, indent=2)
            )
        if self._pending is not None:
            self._launch_pending()
        return results

    def finish(self, final_ckpt: dict[str, Any] | None = None) -> list[EvalResult]:
        """Wait for running/queued jobs, then evaluate `final_ckpt` (replacing a queued one)."""
        if final_ckpt is not None and final_ckpt["global_step"] != self._last_submitted_step:
            self.submit(final_ckpt)
        results = []
        while self._thread is not None or self._pending is not None:
            if self._thread is None:
                self._launch_pending()
            self._thread.join()
            results += self.poll()
        return results

    def is_best(self, res: EvalResult) -> bool:
        return self.best is not None and self.best["step"] == res.step and res.name == "policy"

    # -- internals --------------------------------------------------------------------------

    def _cached_baselines(self) -> dict[str, dict[str, Any]] | None:
        path = self.run / "baselines.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        if data.get("key") != self.baseline_key or not all(b in data for b in BASELINES):
            return None
        return {b: data[b] for b in BASELINES}

    def _launch(self, job: Callable[[], list[Any]]) -> None:
        def run() -> None:
            try:
                self._job_out = job()
            except Exception as e:  # noqa: BLE001 - re-raised by poll() on the training thread
                self._job_out = [e]

        self._thread = threading.Thread(target=run, name="factorio-eval", daemon=True)
        self._thread.start()

    def _launch_pending(self) -> None:
        ckpt, self._pending = self._pending, None
        self._launch(lambda: self._policy_job(ckpt))

    def _baseline_job(self) -> list[Any]:
        return [
            (evaluate(baseline_agent(b, self.seed), self.env, self.episodes, name=b), None)
            for b in BASELINES
        ]

    def _policy_job(self, ckpt: dict[str, Any]) -> list[Any]:
        agent = self.agent_from_checkpoint(ckpt)
        res = evaluate(agent, self.env, self.episodes, step=ckpt["global_step"])
        return [(res, ckpt)]

    def _update_best(self, res: EvalResult, ckpt: dict[str, Any]) -> None:
        if self.best is not None and res.mean_return <= self.best["return"]:
            return
        self.best = res.summary()
        torch.save({**ckpt, "eval": self.best}, self.run / "checkpoint-best.pt")
        (self.run / "best.json").write_text(json.dumps(self.best, indent=2))


def eval_metrics(res: EvalResult, evaluator: Evaluator) -> dict[str, float]:
    """Flat metrics for one policy evaluation, including the gap to each baseline."""
    m = {
        "eval_return": res.mean_return,
        "eval_return_std": res.std_return,
        "eval_length": res.mean_length,
        "eval_errors": res.mean_errors,
        "eval_step": res.step,
    }
    if evaluator.best is not None:
        m["eval_best"] = evaluator.best["return"]
        m["eval_best_step"] = evaluator.best["step"]
    for b, s in evaluator.baselines.items():
        m[f"baseline_{b}"] = s["return"]
        m[f"eval_vs_{b}"] = res.mean_return - s["return"]
    return m


def format_eval(res: EvalResult, evaluator: Evaluator) -> str:
    if res.name in BASELINES:
        return (
            f"baseline {res.name}: return {res.mean_return:.3f} ± {res.std_return:.3f} "
            f"over {len(res.returns)} episodes, errors/episode {res.mean_errors:.1f}"
        )
    parts = [f"eval @ step {res.step}: return {res.mean_return:.3f} ± {res.std_return:.3f}"]
    parts += [f"{b} {s['return']:.3f}" for b, s in evaluator.baselines.items()]
    if evaluator.best is not None:
        parts.append(f"best {evaluator.best['return']:.3f} @ {evaluator.best['step']}")
    line = " | ".join(parts)
    return line + ("  [new best -> checkpoint-best.pt]" if evaluator.is_best(res) else "")


# -- factorio-eval CLI ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="factorio-eval",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("checkpoints", nargs="*", help="factorio-train checkpoints to evaluate")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--baselines", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mock", action="store_true", help="evaluate on MockFactorioEnv")
    parser.add_argument("--container", type=int, default=0, help="Factorio container index")
    parser.add_argument("--env-id", help="FLE task (default: the first checkpoint's)")
    parser.add_argument("--max-episode-steps", type=int, help="default: the first checkpoint's")
    parser.add_argument("--error-penalty", type=float, help="default: the first checkpoint's")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if not args.checkpoints and not args.baselines:
        parser.error("nothing to evaluate")

    from factorio_agent.rl import load_trainer
    from factorio_agent.rl.envs import make_eval_env

    first = (
        torch.load(args.checkpoints[0], map_location="cpu", weights_only=False)["config"]
        if args.checkpoints
        else {}
    )
    env_id = args.env_id or first.get("env_id", "iron_ore_throughput")
    max_steps = args.max_episode_steps or first.get("max_episode_steps", 50)
    penalty = (
        args.error_penalty if args.error_penalty is not None else first.get("error_penalty", 0.0)
    )
    try:
        env = make_eval_env(env_id, args.container, max_steps, penalty, args.mock, args.seed)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(
        f"{env_id}{' (mock)' if args.mock else ''}: {args.episodes} episode(s) each, "
        f"{max_steps} steps max, error penalty {penalty}"
    )

    rows: list[tuple[str, EvalResult]] = []
    try:
        jobs: list[tuple[str, Callable[[], Agent]]] = []
        if args.baselines:
            jobs += [(b, lambda b=b: baseline_agent(b, args.seed)) for b in BASELINES]
        jobs += [
            (p, lambda p=p: load_trainer(p, device=args.device).make_agent())
            for p in args.checkpoints
        ]
        for label, make in jobs:
            res = evaluate(make(), env, args.episodes, name=label)
            rows.append((label, res))
            print(f"  {label}: {res.mean_return:.3f} ± {res.std_return:.3f}", flush=True)
    finally:
        env.close()

    width = max(len(label) for label, _ in rows)
    print(f"\n{'agent'.ljust(width)}  return            length  errors")
    for label, r in sorted(rows, key=lambda x: -x[1].mean_return):
        print(
            f"{label.ljust(width)}  {r.mean_return:8.3f} ± {r.std_return:<6.3f}"
            f"  {r.mean_length:6.1f}  {r.mean_errors:6.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
