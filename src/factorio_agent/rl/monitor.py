"""Monitor training runs from their run directories (metrics.jsonl, checkpoints).

uv run factorio-monitor                      # table of every run under runs/
uv run factorio-monitor --watch 30           # refresh every 30 s
uv run factorio-monitor runs/ppo-gpu         # details and return curve for one run

With `--eval-every`, runs also show no-exploration evaluation returns (see rl/evaluate.py)
next to the random/scripted baselines, the best evaluation (checkpoint-best.pt), and a
warning when the last STALL_EVALS evaluations did not improve on the best.
"""

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from factorio_agent.rl.common import latest_checkpoint
from factorio_agent.rl.evaluate import BASELINES

SPARK = "▁▂▃▄▅▆▇█"
STALL_EVALS = 5  # evaluations without a new best before a run is flagged


@dataclass
class RunSummary:
    name: str
    algo: str
    status: str  # running | stale | finished | starting
    global_step: int
    total_steps: int
    steps_per_sec: float
    episodes: int
    recent_return: float | None  # mean over roughly the last `window` episodes
    best_return: float | None
    age: float  # seconds since the last metrics line
    checkpoint: str
    returns: list[float]  # mean episode return per metrics line that finished episodes
    last: dict[str, float]
    evals: list[tuple[int, float, float]] = field(default_factory=list)  # (step, mean, std)
    best_eval: dict | None = None  # best.json
    baselines: dict[str, dict] = field(default_factory=dict)  # baselines.json, by name

    @property
    def latest_eval(self) -> tuple[int, float, float] | None:
        return self.evals[-1] if self.evals else None

    @property
    def evals_since_best(self) -> int:
        if not self.best_eval:
            return 0
        return sum(1 for step, _, _ in self.evals if step > self.best_eval["step"])

    @property
    def eta(self) -> float | None:
        if self.status != "running" or self.steps_per_sec <= 0:
            return None
        return (self.total_steps - self.global_step) / self.steps_per_sec


def read_metrics(run: Path) -> list[dict[str, float]]:
    path = run / "metrics.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:  # partially written last line
            continue
    return rows


def summarize(run: Path, stale_after: float = 600.0, window: int = 20) -> RunSummary:
    config = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {}
    rows = read_metrics(run)
    last = rows[-1] if rows else {}
    # the newest training line (eval-only lines, e.g. the final evaluation, lack its fields)
    train_last = next((r for r in reversed(rows) if "steps_per_sec" in r), last)
    ended = [r for r in rows if "episode_return" in r]
    returns = [r["episode_return"] for r in ended]
    counts = [int(r.get("episodes", 1)) for r in ended]
    age = time.time() - last.get("time", run.stat().st_mtime)
    ckpt = latest_checkpoint(run)
    if ckpt and ckpt.name == "checkpoint-final.pt":
        status = "finished"
    elif not rows:
        status = "starting" if age < stale_after else "stale"
    else:
        status = "running" if age < stale_after else "stale"
    # weight each line's mean by its episode count, over lines covering the last `window`
    k, covered = 0, 0
    while k < len(ended) and covered < window:
        k += 1
        covered += counts[-k]
    recent_total = sum(returns[-i] * counts[-i] for i in range(1, k + 1))
    return RunSummary(
        name=run.name,
        algo=config.get("algo", "?"),
        status=status,
        global_step=int(last.get("global_step", 0)),
        total_steps=int(config.get("total_steps", 0)),
        steps_per_sec=float(train_last.get("steps_per_sec", 0.0)),
        episodes=sum(counts),
        recent_return=recent_total / covered if covered else None,
        best_return=max(returns) if returns else None,
        age=age,
        checkpoint=ckpt.name if ckpt else "-",
        returns=returns,
        last={k: v for k, v in train_last.items() if not k.startswith(("eval_", "baseline_"))},
        evals=[
            (int(r["eval_step"]), r["eval_return"], r.get("eval_return_std", 0.0))
            for r in rows
            if "eval_return" in r
        ],
        best_eval=_read_json(run / "best.json"),
        baselines={
            k: v for k, v in (_read_json(run / "baselines.json") or {}).items() if k != "key"
        },
    )


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def find_runs(root: Path) -> list[Path]:
    if (root / "config.json").exists():
        return [root]
    return sorted(p.parent for p in root.glob("*/config.json"))


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = int(seconds)
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    if seconds < 86400:
        return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"
    return f"{seconds // 86400}d{seconds % 86400 // 3600:02d}h"


def fmt_num(v: float | None) -> str:
    return "-" if v is None else f"{v:.2f}"


def sparkline(values: list[float], width: int = 60) -> str:
    if not values:
        return ""
    if len(values) > width:  # bucket means
        size = len(values) / width
        values = [
            sum(chunk) / len(chunk)
            for i in range(width)
            if (chunk := values[int(i * size) : int((i + 1) * size)])
        ]
    lo, hi = min(values), max(values)
    span = hi - lo or 1.0
    return "".join(SPARK[int((v - lo) / span * (len(SPARK) - 1))] for v in values)


def table(summaries: list[RunSummary]) -> str:
    header = (
        "run",
        "algo",
        "status",
        "step",
        "progress",
        "steps/s",
        "ETA",
        "episodes",
        "return",
        "best",
        "eval",
        "eval best",
        "rand/script",
        "updated",
        "checkpoint",
    )
    rows = [header]
    for s in summaries:
        progress = f"{100 * s.global_step / s.total_steps:.1f}%" if s.total_steps else "-"
        rows.append(
            (
                s.name,
                s.algo,
                s.status,
                str(s.global_step),
                progress,
                f"{s.steps_per_sec:.3f}",
                fmt_duration(s.eta),
                str(s.episodes),
                fmt_num(s.recent_return),
                fmt_num(s.best_return),
                fmt_num(s.latest_eval[1] if s.latest_eval else None),
                f"{s.best_eval['return']:.2f}@{s.best_eval['step']}" if s.best_eval else "-",
                "/".join(fmt_num(s.baselines.get(b, {}).get("return")) for b in BASELINES)
                if s.baselines
                else "-",
                fmt_duration(s.age) + " ago",
                s.checkpoint,
            )
        )
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)) for r in rows)


def details(s: RunSummary) -> str:
    skip = {"iteration", "global_step", "time"}
    metrics = "\n".join(
        f"  {k:16s} {v:.4g}" if isinstance(v, float) else f"  {k:16s} {v}"
        for k, v in s.last.items()
        if k not in skip
    )
    return (
        f"{table([s])}\n\nlatest metrics:\n{metrics or '  (none yet)'}\n\n"
        f"episode return, mean per log line ({s.episodes} episodes, "
        f"min {fmt_num(min(s.returns, default=None))}, "
        f"max {fmt_num(s.best_return)}):\n  {sparkline(s.returns) or '(no episodes yet)'}"
        f"{eval_details(s)}"
    )


def eval_details(s: RunSummary) -> str:
    if not (s.evals or s.baselines or s.best_eval):
        return ""
    lines = ["", "", "evaluation (no exploration; training returns above include exploration):"]
    if s.baselines:
        lines.append(
            "  baselines: "
            + "   ".join(
                f"{b} {v['return']:.2f} ± {v.get('return_std', 0):.2f}"
                for b, v in s.baselines.items()
            )
        )
    if s.best_eval:
        lines.append(
            f"  best:      {s.best_eval['return']:.2f} at step {s.best_eval['step']} "
            "(checkpoint-best.pt)"
        )
    if s.latest_eval:
        step, ret, std = s.latest_eval
        gaps = "".join(f", {ret - v['return']:+.2f} vs {b}" for b, v in s.baselines.items())
        lines.append(f"  latest:    {ret:.2f} ± {std:.2f} at step {step}{gaps}")
        lines.append(
            f"  eval return per evaluation ({len(s.evals)}):\n  {sparkline([e[1] for e in s.evals])}"
        )
        lines.append(
            "  recent: " + " | ".join(f"{step}: {ret:.2f}" for step, ret, _ in s.evals[-6:])
        )
        if "scripted" in s.baselines:
            beat = s.best_eval and s.best_eval["return"] > s.baselines["scripted"]["return"]
            lines.append(
                "  verdict:   best eval "
                + ("beats" if beat else "has not beaten")
                + " the scripted baseline"
            )
    elif s.baselines:
        lines.append("  (no policy evaluation finished yet)")
    if s.evals_since_best >= STALL_EVALS:
        lines.append(f"  WARNING: {s.evals_since_best} evaluations since the last new best")
    return "\n".join(lines)


def render(root: Path, stale_after: float) -> str:
    runs = find_runs(root)
    if not runs:
        return f"no runs under {root}"
    summaries = [summarize(r, stale_after) for r in runs]
    if len(summaries) == 1:
        return details(summaries[0])
    stale = [s.name for s in summaries if s.status == "stale"]
    stalled = [
        f"{s.name} ({s.evals_since_best})" for s in summaries if s.evals_since_best >= STALL_EVALS
    ]
    out = table(summaries)
    if stale:
        out += f"\n\nWARNING: no progress for {fmt_duration(stale_after)}: {', '.join(stale)}"
    if stalled:
        out += f"\n\nWARNING: no new best in the last evaluations: {', '.join(stalled)}"
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="factorio-monitor",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("path", nargs="?", default="runs", help="runs root or one run directory")
    parser.add_argument("--watch", type=float, default=0, help="refresh every N seconds")
    parser.add_argument(
        "--stale-after",
        type=float,
        default=600,
        help="seconds without a metrics line before a run is flagged stale",
    )
    args = parser.parse_args(argv)
    root = Path(args.path)
    if not args.watch:
        print(render(root, args.stale_after))
        return 0
    try:
        while True:
            print("\033[2J\033[H" + time.strftime("%Y-%m-%d %H:%M:%S"), end="\n\n")
            print(render(root, args.stale_after), flush=True)
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
