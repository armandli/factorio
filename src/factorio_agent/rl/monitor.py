"""Monitor training runs from their run directories (metrics.jsonl, checkpoints).

uv run factorio-monitor                      # table of every run under runs/
uv run factorio-monitor --watch 30           # refresh every 30 s
uv run factorio-monitor runs/ppo-gpu         # details and return curve for one run
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from factorio_agent.rl.common import latest_checkpoint

SPARK = "▁▂▃▄▅▆▇█"


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
        steps_per_sec=float(last.get("steps_per_sec", 0.0)),
        episodes=sum(counts),
        recent_return=recent_total / covered if covered else None,
        best_return=max(returns) if returns else None,
        age=age,
        checkpoint=ckpt.name if ckpt else "-",
        returns=returns,
        last=last,
    )


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
    )


def render(root: Path, stale_after: float) -> str:
    runs = find_runs(root)
    if not runs:
        return f"no runs under {root}"
    summaries = [summarize(r, stale_after) for r in runs]
    if len(summaries) == 1:
        return details(summaries[0])
    stale = [s.name for s in summaries if s.status == "stale"]
    warning = f"\n\nWARNING: no progress for {fmt_duration(stale_after)}: {', '.join(stale)}"
    return table(summaries) + (warning if stale else "")


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
