"""Shared training infrastructure: config, AMP helper, logging, checkpoints, trainer base."""

import contextlib
import json
import random
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar

import torch
from torch import nn

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # tensorboard is in the optional `train` dependency group
    SummaryWriter = None

from factorio_agent.agents import Agent
from factorio_agent.rl.buffers import ReplayBuffer
from factorio_agent.rl.envs import MacroEnv, VecMacroEnv, make_eval_env
from factorio_agent.rl.evaluate import Evaluator, eval_metrics, format_eval


@dataclass
class TrainConfig:
    """Fields shared by every algorithm. Subclasses add algorithm hyperparameters; the
    `factorio-train` CLI exposes every field as a `--flag`."""

    # environment
    env_id: str = "iron_ore_throughput"
    mock: bool = False  # train on MockFactorioEnv instead of a Factorio server
    num_envs: int = 1  # one Factorio container per env
    first_container: int = 0  # env i uses container first_container + i
    max_episode_steps: int = 50
    error_penalty: float = 0.0  # subtracted from the reward when a program errors
    # optimisation
    total_steps: int = 10_000  # env steps summed over all envs
    gamma: float = 0.99
    lr: float = 3e-4
    max_grad_norm: float = 1.0
    amp: bool = True  # autocast (bf16 on CPU/MPS, fp16 + GradScaler on CUDA)
    device: str = "auto"
    seed: int = 0
    # bookkeeping
    run_dir: str = "runs"
    run_name: str = ""  # default: <algo>-<timestamp>
    log_every: int = 1  # iterations between log lines
    checkpoint_every: int = 0  # env steps between checkpoints (0: only at the end)
    keep_checkpoints: int = 3  # periodic checkpoints kept (final one always kept)
    tensorboard: bool = True  # also log to TensorBoard if it is installed
    # evaluation (see rl/evaluate.py): no-exploration episodes on a dedicated env
    eval_every: int = 0  # env steps between evaluations (0: off)
    eval_episodes: int = 3  # episodes per evaluation
    eval_container: int = -1  # Factorio container for evaluation (-1: first_container + num_envs)
    eval_baselines: bool = True  # also evaluate the random and scripted agents once per run

    def resolved_eval_container(self) -> int:
        return (
            self.eval_container
            if self.eval_container >= 0
            else self.first_container + self.num_envs
        )


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Amp:
    """Mixed precision for one optimizer. Losses should be computed from `.float()` outputs."""

    def __init__(self, device: torch.device, enabled: bool):
        self.device = device
        self.enabled = enabled
        self.dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
        self.scaler = torch.amp.GradScaler(device.type, enabled=enabled and device.type == "cuda")

    def autocast(self) -> contextlib.AbstractContextManager:
        return torch.autocast(self.device.type, dtype=self.dtype, enabled=self.enabled)

    def step(
        self, loss: torch.Tensor, optimizer: torch.optim.Optimizer, params, max_grad_norm: float
    ) -> float:
        """Backward, clip, step. Returns the pre-clip gradient norm."""
        optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(params, max_grad_norm)
        self.scaler.step(optimizer)
        self.scaler.update()
        return float(grad_norm)


class Trainer(ABC):
    """Base for RL algorithms.

    A subclass builds its networks in `__init__`, implements `train_iteration` (collect some
    experience, update, advance `self.global_step`), `make_agent` (an `Agent` for playing
    with the current weights) and `modules` (what to checkpoint). `envs` may be None when
    the trainer is only built to load a checkpoint for inference.
    """

    name: ClassVar[str]
    config_cls: ClassVar[type[TrainConfig]] = TrainConfig

    def __init__(self, cfg: TrainConfig, envs: VecMacroEnv | None):
        self.cfg = cfg
        self.envs = envs
        self.device = resolve_device(cfg.device)
        self.global_step = 0
        self.iteration = 0
        if envs is not None:  # not for inference/eval copies, which must not reseed globally
            random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)

    @abstractmethod
    def train_iteration(self) -> dict[str, float]: ...

    @abstractmethod
    def make_agent(self) -> Agent: ...

    @abstractmethod
    def modules(self) -> dict[str, Any]:
        """Objects with state_dict()/load_state_dict(): networks, optimizers, ..."""

    def train(self, eval_env: MacroEnv | None = None) -> Path:
        """Train until `total_steps` env steps; returns the final checkpoint path.

        With `eval_every`, the policy is evaluated on `eval_env` (built from the config if not
        given) alongside training, the best snapshot is kept as `checkpoint-best.pt`, and the
        final weights get one last evaluation.

        If training raises (e.g. a Factorio server stops responding), a checkpoint is saved
        before re-raising so the run can be resumed with `--resume`.
        """
        assert self.envs is not None, "trainer was built without envs"
        cfg = self.cfg
        run = Path(cfg.run_dir) / (cfg.run_name or f"{self.name}-{time.strftime('%Y%m%d-%H%M%S')}")
        run.mkdir(parents=True, exist_ok=True)
        (run / "config.json").write_text(json.dumps({"algo": self.name, **asdict(cfg)}, indent=2))
        writer = SummaryWriter(str(run / "tb")) if cfg.tensorboard and SummaryWriter else None
        episodes: list[float] = []
        start, start_step = time.time(), self.global_step
        last_checkpoint = last_eval = self.global_step
        own_eval_env = False
        if cfg.eval_every and eval_env is None:
            eval_env, own_eval_env = self._make_eval_env(), True
        evaluator = self._make_evaluator(eval_env, run) if cfg.eval_every else None
        try:
            with open(run / "metrics.jsonl", "a") as log:
                while self.global_step < cfg.total_steps:
                    metrics = self.train_iteration()
                    self.iteration += 1
                    stats = self.envs.pop_episode_stats()
                    episodes += [s.episode_return for s in stats]
                    if stats:
                        n = len(stats)
                        metrics["episodes"] = n
                        metrics["episode_return"] = sum(s.episode_return for s in stats) / n
                        metrics["episode_length"] = sum(s.length for s in stats) / n
                        metrics["episode_errors"] = sum(s.errors for s in stats) / n
                    metrics["steps_per_sec"] = (self.global_step - start_step) / (
                        time.time() - start
                    )
                    if evaluator:
                        if self.global_step - last_eval >= cfg.eval_every:
                            evaluator.submit(self.checkpoint_dict(include_buffers=False))
                            last_eval = self.global_step
                        metrics.update(self._eval_results(evaluator, evaluator.poll(), writer))
                    self._record(metrics, log, writer, echo=self.iteration % cfg.log_every == 0)
                    if cfg.checkpoint_every and (
                        self.global_step - last_checkpoint >= cfg.checkpoint_every
                    ):
                        self._checkpoint(run)
                        last_checkpoint = self.global_step
                if evaluator:
                    print("final evaluation...", flush=True)
                    final = evaluator.finish(self.checkpoint_dict(include_buffers=False))
                    metrics = self._eval_results(evaluator, final, writer)
                    if metrics:
                        self._record(metrics, log, writer, echo=False)
        except BaseException:
            print(f"training stopped at step {self.global_step}; saving a checkpoint", flush=True)
            self._checkpoint(run)
            raise
        finally:
            if writer:
                writer.close()
            if own_eval_env:
                eval_env.close()
        path = run / "checkpoint-final.pt"
        self.save(path)
        recent = episodes[-20:]
        if recent:
            print(
                f"done: {len(episodes)} episodes, mean return (last {len(recent)}) "
                f"{sum(recent) / len(recent):.3f}"
            )
        if evaluator and evaluator.best:
            print(
                f"best eval return {evaluator.best['return']:.3f} at step "
                f"{evaluator.best['step']}: {run / 'checkpoint-best.pt'}"
            )
        print(f"checkpoint: {path}")
        return path

    def _record(self, metrics: dict[str, float], log, writer, echo: bool) -> None:
        """One metrics.jsonl line (+ TensorBoard scalars, + a console line if `echo`)."""
        if writer:
            for k, v in metrics.items():
                if not k.startswith(("eval_", "baseline_")):  # logged by _eval_results
                    writer.add_scalar(k, v, self.global_step)
        metrics.update(iteration=self.iteration, global_step=self.global_step, time=time.time())
        log.write(json.dumps(metrics) + "\n")
        log.flush()
        if echo:
            print(format_metrics(metrics, skip={"time"}), flush=True)

    def _eval_results(self, evaluator: Evaluator, results, writer) -> dict[str, float]:
        """Print finished evaluations, log them to TensorBoard at the evaluated step, and
        return the newest policy result's metrics for metrics.jsonl."""
        metrics: dict[str, float] = {}
        for res in results:
            print(format_eval(res, evaluator), flush=True)
            if res.name != "policy":
                continue
            metrics = eval_metrics(res, evaluator)
            if writer:
                step = res.step
                writer.add_scalar("eval/return", res.mean_return, step)
                writer.add_scalar("eval/return_std", res.std_return, step)
                writer.add_scalar("eval/errors", res.mean_errors, step)
                writer.add_scalar("eval/best_return", evaluator.best["return"], step)
                writer.add_scalars(
                    "eval/vs_baselines",
                    {"policy": res.mean_return}
                    | {b: s["return"] for b, s in evaluator.baselines.items()},
                    step,
                )
                for b, s in evaluator.baselines.items():
                    writer.add_scalar(f"eval/policy_minus_{b}", res.mean_return - s["return"], step)
        return metrics

    def _make_eval_env(self) -> MacroEnv:
        cfg = self.cfg
        return make_eval_env(
            cfg.env_id,
            cfg.resolved_eval_container(),
            cfg.max_episode_steps,
            cfg.error_penalty,
            mock=cfg.mock,
            seed=cfg.seed,
        )

    def _make_evaluator(self, env: MacroEnv, run: Path) -> Evaluator:
        cfg = self.cfg
        # a weights-only copy of this trainer for the eval thread, built here (main thread)
        # and reloaded for each snapshot
        player = type(self)(cfg, None)

        def agent_from_checkpoint(ckpt: dict[str, Any]) -> Agent:
            player.load(ckpt)
            return player.make_agent()

        return Evaluator(
            env,
            run,
            cfg.eval_episodes,
            agent_from_checkpoint,
            baselines=cfg.eval_baselines,
            baseline_key={
                "env_id": cfg.env_id,
                "mock": cfg.mock,
                "max_episode_steps": cfg.max_episode_steps,
                "error_penalty": cfg.error_penalty,
                "episodes": cfg.eval_episodes,
                "seed": cfg.seed,
            },
            seed=cfg.seed,
        )

    def _checkpoint(self, run: Path) -> None:
        self.save(run / f"checkpoint-{self.global_step}.pt")
        for old in periodic_checkpoints(run)[: -self.cfg.keep_checkpoints or None]:
            old.unlink()

    def checkpoint_dict(self, include_buffers: bool = True) -> dict[str, Any]:
        """What `save` writes. Without buffers it is small enough to snapshot for evaluation
        (and the result still loads with `load_trainer` / `load_agent`)."""
        return {
            "algo": self.name,
            "config": asdict(self.cfg),
            "global_step": self.global_step,
            "iteration": self.iteration,
            "state": {
                k: m.state_dict()
                for k, m in self.modules().items()
                if include_buffers or not isinstance(m, ReplayBuffer)
            },
        }

    def save(self, path: str | Path) -> None:
        torch.save(self.checkpoint_dict(), path)

    def load(self, ckpt: dict[str, Any]) -> None:
        for k, m in self.modules().items():
            if k in ckpt["state"]:
                m.load_state_dict(ckpt["state"][k])
        self.global_step = ckpt["global_step"]
        self.iteration = ckpt["iteration"]


def periodic_checkpoints(run: str | Path) -> list[Path]:
    """`checkpoint-<step>.pt` files in a run directory, oldest first (excludes final)."""
    paths = [p for p in Path(run).glob("checkpoint-*.pt") if p.stem.split("-")[1].isdigit()]
    return sorted(paths, key=lambda p: int(p.stem.split("-")[1]))


def latest_checkpoint(run: str | Path) -> Path | None:
    """The final checkpoint if training finished, else the newest periodic one."""
    final = Path(run) / "checkpoint-final.pt"
    if final.exists():
        return final
    periodic = periodic_checkpoints(run)
    return periodic[-1] if periodic else None


def format_metrics(metrics: dict[str, float], skip: set[str] = frozenset()) -> str:
    def fmt(v: float) -> str:
        return f"{v:.4g}" if isinstance(v, float) else str(v)

    return "  ".join(f"{k}={fmt(v)}" for k, v in metrics.items() if k not in skip)
