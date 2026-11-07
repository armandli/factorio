"""Offline tests for the RL training code, on MockFactorioEnv (no Factorio server needed)."""

import dataclasses

import pytest
import torch

from factorio_agent.actions import MACRO_ACTIONS, NUM_ACTIONS
from factorio_agent.features import FEATURE_DIM
from factorio_agent.rl import ALGORITHMS, load_agent, load_trainer, make_vec_env
from factorio_agent.rl.buffers import NStepAccumulator, ReplayBuffer, RolloutBuffer
from factorio_agent.rl.common import latest_checkpoint, periodic_checkpoints
from factorio_agent.rl.mock_env import MockFactorioEnv
from factorio_agent.rl.monitor import render, sparkline, summarize
from factorio_agent.rl.search import MCGS
from factorio_agent.rl.train import main as train_main

ACTION = {a.name: i for i, a in enumerate(MACRO_ACTIONS)}

# tiny settings so every algorithm runs a few updates in well under a second
SMOKE = {
    "ppo": {"num_steps": 8, "minibatch_size": 8, "update_epochs": 2},
    "dqn": {"learning_starts": 16, "batch_size": 8},
    "mcgs": {"learning_starts": 16, "batch_size": 8, "num_simulations": 8, "gradient_steps": 1},
}


def test_mock_env_dynamics():
    venv = make_vec_env(mock=True, num_envs=1, max_episode_steps=3)
    venv.reset()
    step = venv.step([ACTION["place_iron_drill"]])
    assert step.reward[0] == 1.0 and not step.error[0]
    step = venv.step([ACTION["craft_gear_wheel"]])  # no iron plates
    assert step.error[0] and step.reward[0] == 1.0
    step = venv.step([ACTION["wait"]])
    assert step.truncated[0]
    assert not torch.equal(step.obs[0], step.next_obs[0])  # obs was auto-reset
    [stats] = venv.pop_episode_stats()
    assert stats.length == 3 and stats.episode_return == 3.0 and stats.errors == 1
    venv.close()


def test_rollout_buffer_gae_matches_discounted_return():
    buf = RolloutBuffer(num_steps=3, num_envs=1, obs_dim=FEATURE_DIM)
    for t in range(3):
        buf.add(
            torch.zeros(1, FEATURE_DIM),
            torch.zeros(1),
            torch.zeros(1),
            torch.ones(1),
            torch.tensor([float(t == 2)]),
            torch.zeros(1),
        )
    buf.compute_gae(torch.tensor([100.0]), gamma=0.5, gae_lambda=1.0)
    # terminal at t=2, so the last value is ignored: returns are 1.75, 1.5, 1
    assert torch.allclose(buf.returns[:, 0], torch.tensor([1.75, 1.5, 1.0]))


def test_nstep_accumulator():
    acc = NStepAccumulator(n=2, gamma=0.5)
    s = [torch.full((FEATURE_DIM,), float(i)) for i in range(4)]
    assert acc.push(s[0], 0, 1.0, s[1], False, False) == []
    [tr] = acc.push(s[1], 1, 2.0, s[2], False, False)
    assert tr["reward"] == 2.0 and tr["discount"] == 0.25 and torch.equal(tr["next_obs"], s[2])
    out = acc.push(s[2], 2, 4.0, s[3], True, True)  # terminal: flush both windows
    assert [t["reward"] for t in out] == [4.0, 4.0] and all(t["done"] == 1.0 for t in out)
    assert [t["discount"] for t in out] == [0.25, 0.5]


def test_replay_buffer_wraps():
    buf = ReplayBuffer(4, obs=(2,), action=())
    for i in range(6):
        buf.add(obs=torch.full((2,), float(i)), action=i)
    assert len(buf) == 4
    batch = buf.sample(16)
    assert batch["action"].dtype == torch.long
    assert set(batch["action"].tolist()) <= {2, 3, 4, 5}


class AdditiveModel:
    """Exact model where actions commute: each action adds a unit vector to the state."""

    def evaluate(self, state):
        return torch.full((NUM_ACTIONS,), 1.0 / NUM_ACTIONS), 0.0

    def step(self, state, action):
        nxt = state.clone()
        nxt[action] += 1.0
        return nxt, float(action == 3), False


def test_mcgs_finds_reward():
    result = MCGS(AdditiveModel(), NUM_ACTIONS, num_simulations=100, gamma=0.9).search(
        torch.zeros(FEATURE_DIM)
    )
    assert torch.isclose(result.policy.sum(), torch.tensor(1.0))
    assert int(result.policy.argmax()) == 3


class RewardFreeAdditiveModel(AdditiveModel):
    def step(self, state, action):
        nxt, _, done = super().step(state, action)
        return nxt, 0.0, done


def test_mcgs_merges_transpositions():
    # with no reward the search spreads out, and a+b / b+a reach the same state; a tree
    # would create one node per simulation (+ root)
    result = MCGS(RewardFreeAdditiveModel(), NUM_ACTIONS, num_simulations=200).search(
        torch.zeros(FEATURE_DIM)
    )
    assert result.num_nodes < 150


class SelfLoopModel(AdditiveModel):
    def step(self, state, action):
        return state.clone(), 0.0, False  # every action returns to the same state


def test_mcgs_handles_cycles():
    result = MCGS(SelfLoopModel(), NUM_ACTIONS, num_simulations=50).search(
        torch.zeros(FEATURE_DIM), add_noise=True
    )
    assert result.num_nodes == 1
    assert torch.isclose(result.policy.sum(), torch.tensor(1.0))


@pytest.mark.parametrize("algo", sorted(ALGORITHMS))
def test_trainers_run_and_checkpoint(algo, tmp_path):
    cls = ALGORITHMS[algo]
    cfg = cls.config_cls(
        mock=True,
        num_envs=2,
        total_steps=48,
        max_episode_steps=10,
        device="cpu",
        run_dir=str(tmp_path),
        run_name=algo,
        **SMOKE[algo],
    )
    envs = make_vec_env(mock=True, num_envs=2, max_episode_steps=10)
    trainer = cls(cfg, envs)
    metrics = trainer.train_iteration()
    assert all(isinstance(v, float) for v in metrics.values())
    path = trainer.train()
    envs.close()
    assert trainer.global_step >= 48
    assert (tmp_path / algo / "metrics.jsonl").read_text().count("\n") >= 2

    restored = load_trainer(path, device="cpu")
    assert restored.global_step == trainer.global_step
    agent = load_agent(path)
    env = MockFactorioEnv(seed=0)
    obs, _ = env.reset()
    assert agent.act(obs) in MACRO_ACTIONS


def test_train_cli_on_mock(tmp_path, capsys):
    rc = train_main(
        [
            "ppo",
            "--mock",
            "--total-steps",
            "16",
            "--num-steps",
            "8",
            "--minibatch-size",
            "8",
            "--no-amp",
            "--device",
            "cpu",
            "--run-dir",
            str(tmp_path),
            "--run-name",
            "cli",
        ]
    )
    assert rc == 0
    assert (tmp_path / "cli" / "checkpoint-final.pt").exists()
    assert "checkpoint:" in capsys.readouterr().out


def test_replay_buffer_state_round_trip():
    buf = ReplayBuffer(8, obs=(2,), action=())
    for i in range(5):
        buf.add(obs=torch.full((2,), float(i)), action=i)
    restored = ReplayBuffer(8, obs=(2,), action=())
    restored.load_state_dict(buf.state_dict())
    assert len(restored) == 5 and restored.pos == buf.pos
    assert torch.equal(restored.data["obs"][:5], buf.data["obs"][:5])


class Boom(RuntimeError):
    pass


def test_crash_saves_checkpoint_and_resumes(tmp_path, monkeypatch):
    cls = ALGORITHMS["dqn"]
    cfg = cls.config_cls(
        mock=True,
        total_steps=40,
        device="cpu",
        run_dir=str(tmp_path),
        run_name="crash",
        checkpoint_every=5,
        keep_checkpoints=2,
        **SMOKE["dqn"],
    )
    envs = make_vec_env(mock=True)
    trainer = cls(cfg, envs)
    real_iteration = trainer.train_iteration

    def flaky():
        if trainer.global_step == 23:
            raise Boom("server went away")
        return real_iteration()

    monkeypatch.setattr(trainer, "train_iteration", flaky)
    with pytest.raises(Boom):
        trainer.train()
    run = tmp_path / "crash"
    ckpt = latest_checkpoint(run)
    assert ckpt.name == "checkpoint-23.pt"
    assert len(periodic_checkpoints(run)) == 2  # older ones pruned

    resumed = load_trainer(ckpt, envs, **dataclasses.asdict(cfg))
    assert resumed.global_step == 23 and len(resumed.buffer) == len(trainer.buffer)
    resumed.train()
    assert latest_checkpoint(run).name == "checkpoint-final.pt"
    assert summarize(run).status == "finished"
    envs.close()


def test_monitor_summarizes_runs(tmp_path):
    for algo, steps in (("ppo", 16), ("mcgs", 24)):
        cls = ALGORITHMS[algo]
        cfg = cls.config_cls(
            mock=True,
            num_envs=2,
            total_steps=steps,
            max_episode_steps=4,
            device="cpu",
            run_dir=str(tmp_path),
            run_name=algo,
            **SMOKE[algo],
        )
        envs = make_vec_env(mock=True, num_envs=2, max_episode_steps=4)
        cls(cfg, envs).train()
        envs.close()
    s = summarize(tmp_path / "ppo")
    assert s.algo == "ppo" and s.status == "finished" and s.global_step == 16
    assert s.episodes == 4 and s.recent_return is not None
    out = render(tmp_path, stale_after=600)
    assert "ppo" in out and "mcgs" in out and "finished" in out
    assert "episode return" in render(tmp_path / "mcgs", stale_after=600)
    assert sparkline([0.0, 1.0, 2.0]) == "▁▄█"


def test_evaluate_baselines_on_mock():
    from factorio_agent.agents import ScriptedAgent
    from factorio_agent.rl import evaluate, make_eval_env

    env = make_eval_env(mock=True, max_episode_steps=5)
    res = evaluate(ScriptedAgent(), env, episodes=2, name="scripted")
    assert res.lengths == [5, 5] and len(res.returns) == 2
    assert res.mean_return == sum(res.returns) / 2 and res.std_return >= 0


def test_training_with_eval_keeps_best_checkpoint(tmp_path, capsys):
    import json

    from factorio_agent.rl.evaluate import main as eval_main

    cls = ALGORITHMS["dqn"]
    cfg = cls.config_cls(
        mock=True,
        total_steps=64,
        max_episode_steps=8,
        device="cpu",
        run_dir=str(tmp_path),
        run_name="ev",
        eval_every=16,
        eval_episodes=1,
        **SMOKE["dqn"],
    )
    envs = make_vec_env(mock=True, max_episode_steps=8)
    cls(cfg, envs).train()
    run = tmp_path / "ev"
    out = capsys.readouterr().out
    assert "baseline random" in out and "baseline scripted" in out and "eval @ step" in out

    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    evals = [r for r in rows if "eval_return" in r]
    steps = [r["eval_step"] for r in evals]
    assert steps == sorted(set(steps)) and steps[-1] == 64  # final weights evaluated once
    best = json.loads((run / "best.json").read_text())
    assert best["return"] == max(r["eval_return"] for r in evals)
    assert all("baseline_scripted" in r and "eval_vs_scripted" in r for r in evals)

    ckpt = torch.load(run / "checkpoint-best.pt", weights_only=False)
    assert ckpt["global_step"] == best["step"] and "buffer" not in ckpt["state"]
    assert load_agent(run / "checkpoint-best.pt").act(MockFactorioEnv().reset()[0]) in MACRO_ACTIONS

    s = summarize(run)
    assert s.best_eval["step"] == best["step"] and set(s.baselines) == {"random", "scripted"}
    assert "scripted baseline" in render(run, stale_after=600)

    # resuming reuses the cached baselines and competes against the earlier best
    cfg2 = dataclasses.replace(cfg, total_steps=80)
    resumed = load_trainer(run / "checkpoint-final.pt", envs, **dataclasses.asdict(cfg2))
    resumed.train()
    assert "baseline random" not in capsys.readouterr().out
    assert json.loads((run / "best.json").read_text())["return"] >= best["return"]
    envs.close()

    assert eval_main([str(run / "checkpoint-best.pt"), "--mock", "--episodes", "1"]) == 0
    table = capsys.readouterr().out
    assert "scripted" in table and "checkpoint-best.pt" in table


def test_evaluator_surfaces_worker_errors(tmp_path):
    from factorio_agent.rl import Evaluator, make_eval_env

    def broken(ckpt):
        raise ConnectionError("eval server went away")

    ev = Evaluator(make_eval_env(mock=True), tmp_path, 1, broken, baselines=False)
    ev.submit({"global_step": 1, "state": {}})
    with pytest.raises(RuntimeError, match="evaluation failed"):
        ev.finish()
