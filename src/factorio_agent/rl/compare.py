"""Compare runs by sample efficiency: how much return they get per env step.

uv run factorio-compare runs/ppo-X runs/dqn-X runs/mcgs-X       # table
uv run factorio-compare runs/*-20261111-075642 --json            # machine-readable, with winner
uv run factorio-compare runs/*-X --budget 3000                   # compare up to 3000 env steps

Every run is cut at a common env-step budget (default: the smallest step count any of them
reached), so a run isn't favoured for having taken more steps. Within the budget:

- eval_auc:  area under the no-exploration eval-return curve divided by the budget, i.e. the
             mean eval return over training (step-function interpolation, starting from the
             first evaluation). Rewards learning fast *and* staying good.
- train_auc: the same over the exploring policy's episode returns (more points, noisier).
- best_eval: best eval return within the budget; last_eval: the newest one.

By default (`--metric auto`) the winner is the run with the highest eval_auc; if fewer than
`--min-evals` evaluations fall inside the budget for some run, train_auc decides instead.
`--metric train` ranks by train_auc regardless. Prefer it when the eval modes differ in kind:
greedy evaluation (DQN, MCGS) can lock into repeating one action in FLE's deterministic game
(40+ errors in a 50-step episode) while PPO's sampled evaluation can't, so eval_auc then
measures how each eval mode fails rather than how fast each algorithm learns.
"""

import argparse
import json
import sys
from pathlib import Path

from factorio_agent.rl.monitor import _read_json, find_runs, read_metrics


def _step_auc(points: list[tuple[int, float]], budget: int) -> float | None:
    """Mean of a step function (value held until the next point) over [first point, budget]."""
    pts = sorted((s, v) for s, v in points if s <= budget)
    if not pts:
        return None
    start = pts[0][0]
    if budget <= start:
        return pts[0][1]
    area = 0.0
    for (s, v), nxt in zip(pts, pts[1:] + [(budget, None)]):
        area += v * (nxt[0] - s)
    return area / (budget - start)


def _binned(points: list[tuple[int, float]], budget: int, bins: int) -> list[float | None]:
    out: list[list[float]] = [[] for _ in range(bins)]
    for s, v in points:
        if s <= budget and budget > 0:
            out[min(bins - 1, s * bins // budget)].append(v)
    return [sum(b) / len(b) if b else None for b in out]


def run_points(run: Path) -> dict:
    rows = read_metrics(run)
    config = _read_json(run / "config.json") or {}
    evals = sorted(
        {int(r["eval_step"]): r["eval_return"] for r in rows if "eval_return" in r}.items()
    )
    train = [(int(r["global_step"]), r["episode_return"]) for r in rows if "episode_return" in r]
    steps = max((int(r.get("global_step", 0)) for r in rows), default=0)
    sps = next((r["steps_per_sec"] for r in reversed(rows) if "steps_per_sec" in r), 0.0)
    return {
        "run": str(run),
        "algo": config.get("algo", "?"),
        "steps": steps,
        "steps_per_sec": sps,
        "evals": evals,
        "train": train,
        "baselines": {
            k: v["return"]
            for k, v in (_read_json(run / "baselines.json") or {}).items()
            if k != "key" and isinstance(v, dict)
        },
    }


def compare(
    runs: list[Path],
    budget: int | None = None,
    min_evals: int = 2,
    bins: int = 6,
    metric: str = "auto",
) -> dict:
    data = [run_points(r) for r in runs]
    data = [d for d in data if d["steps"] > 0]
    if not data:
        return {"budget": 0, "runs": [], "winner": None, "decided_by": None}
    budget = budget or min(d["steps"] for d in data)
    for d in data:
        evals = [(s, v) for s, v in d["evals"] if s <= budget]
        d["n_evals"] = len(evals)
        d["eval_auc"] = _step_auc(evals, budget)
        d["train_auc"] = _step_auc(d["train"], budget)
        d["best_eval"] = max((v for _, v in evals), default=None)
        d["last_eval"] = evals[-1][1] if evals else None
        d["train_curve"] = _binned(d["train"], budget, bins)
        d["eval_curve"] = _binned(evals, budget, bins)
    if metric == "auto":
        use_eval = all(d["n_evals"] >= min_evals for d in data)
        key = "eval_auc" if use_eval else "train_auc"
    else:
        key = f"{metric}_auc"
    ranked = sorted(
        data, key=lambda d: d[key] if d[key] is not None else float("-inf"), reverse=True
    )
    for d in data:
        del d["evals"], d["train"]
    return {
        "budget": budget,
        "decided_by": key,
        "winner": ranked[0]["run"] if ranked[0][key] is not None else None,
        "runs": ranked,
    }


def _f(v: float | None, w: int = 8) -> str:
    return f"{v:{w}.2f}" if v is not None else f"{'-':>{w}}"


def render(result: dict) -> str:
    lines = [
        (
            f"sample-efficiency comparison over the first {result['budget']} env steps "
            f"(ranked by {result['decided_by']})"
        ),
        "",
    ]
    lines.append(
        f"{'run':<28}{'steps':>8}{'sps':>7}{'evals':>6}{'eval_auc':>9}{'best':>9}"
        f"{'last':>9}{'trn_auc':>9}  {'scripted':>8}"
    )
    for d in result["runs"]:
        lines.append(
            f"{Path(d['run']).name:<28}{d['steps']:>8}{d['steps_per_sec']:>7.2f}{d['n_evals']:>6}"
            f"{_f(d['eval_auc'], 9)}{_f(d['best_eval'], 9)}{_f(d['last_eval'], 9)}"
            f"{_f(d['train_auc'], 9)}  {_f(d['baselines'].get('scripted'))}"
        )
    lines += ["", "train return by step bin (exploring policy):"]
    for d in result["runs"]:
        lines.append(f"  {Path(d['run']).name:<26}" + " ".join(_f(v, 7) for v in d["train_curve"]))
    lines += ["eval return by step bin:"]
    for d in result["runs"]:
        lines.append(f"  {Path(d['run']).name:<26}" + " ".join(_f(v, 7) for v in d["eval_curve"]))
    if result["winner"]:
        lines += ["", f"most sample-efficient: {Path(result['winner']).name}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("runs", nargs="+", type=Path, help="run directories (or a parent of them)")
    p.add_argument("--budget", type=int, default=None, help="env steps to compare over")
    p.add_argument("--min-evals", type=int, default=2)
    p.add_argument(
        "--metric",
        choices=("auto", "eval", "train"),
        default="auto",
        help="rank by eval_auc, train_auc, or eval_auc once every run has --min-evals (auto)",
    )
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    runs = [r for root in args.runs for r in find_runs(root)]
    result = compare(runs, args.budget, args.min_evals, metric=args.metric)
    print(json.dumps(result, indent=2) if args.json else render(result))
    return 0 if result["winner"] else 1


if __name__ == "__main__":
    sys.exit(main())
