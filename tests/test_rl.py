"""Offline tests for the RL training code, on MockFactorioEnv (no Factorio server needed)."""

import dataclasses

import pytest
import torch

from factorio_agent.actions import MACRO_ACTIONS, NUM_ACTIONS
from factorio_agent.features import FEATURE_DIM
from factorio_agent.rl import ALGORITHMS, load_agent, load_trainer, make_vec_env
from factorio_agent.rl.buffers import (
    NStepAccumulator,
    PrioritizedReplayBuffer,
    ReplayBuffer,
    RolloutBuffer,
    SumTree,
)
from factorio_agent.rl.common import latest_checkpoint, periodic_checkpoints
from factorio_agent.rl.mock_env import MockFactorioEnv
from factorio_agent.rl.monitor import render, sparkline, summarize
from factorio_agent.rl.networks import NoisyLinear, QNetwork, categorical_projection
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


@pytest.mark.parametrize("capacity", [1, 3, 4, 5])
def test_sum_tree_finds_proportional_leaf(capacity):
    tree = SumTree(capacity)
    priorities = torch.arange(1.0, capacity + 1.0)
    tree.update(torch.arange(capacity), priorities)
    assert tree.total == priorities.sum().item()
    edges = priorities.cumsum(0)
    mids = edges - priorities / 2
    assert tree.find(mids).tolist() == list(range(capacity))
    tree.update(torch.tensor([0, 0]), torch.tensor([9.0, 2.0]))  # duplicates: last one wins
    assert tree.total == priorities.sum().item() + 1.0


def test_sum_tree_never_returns_an_empty_leaf():
    tree = SumTree(5)  # padded to 8 leaves
    tree.update(torch.arange(3), torch.ones(3))
    assert tree.find(torch.tensor([3.0, 3.5])).tolist() == [2, 2]


def make_per(capacity=8, n=4, omega=1.0):
    buf = PrioritizedReplayBuffer(capacity, omega, obs=(1,), action=())
    for i in range(n):
        buf.add(obs=torch.tensor([float(i)]), action=i)
    return buf


def test_per_samples_in_proportion_and_corrects_with_weights():
    buf = make_per()
    buf.update_priorities(torch.arange(4), torch.tensor([1.0, 1.0, 1.0, 7.0]))
    batch = buf.sample(1000, beta=1.0)
    counts = batch["index"].bincount(minlength=4).float() / 1000
    assert torch.allclose(counts, torch.tensor([0.1, 0.1, 0.1, 0.7]), atol=0.03)
    assert torch.equal(batch["action"], batch["index"])
    # weights undo the sampling bias: the common transition gets 1/7 of a rare one's weight
    w = batch["weight"]
    assert w.max() == 1.0
    assert torch.allclose(w[batch["index"] == 3], torch.tensor(1 / 7), atol=1e-3)
    assert torch.allclose(w[batch["index"] == 0], torch.tensor(1.0), atol=1e-3)
    assert torch.all(buf.sample(16, beta=0.0)["weight"] == 1.0)


def test_per_new_transitions_get_max_priority_and_overwrites_replace_priority():
    buf = make_per(capacity=4)
    buf.update_priorities(torch.arange(4), torch.tensor([3.0, 1.0, 1.0, 1.0]))
    assert buf.max_priority == pytest.approx(3.0, abs=1e-5)
    buf.update_priorities(torch.tensor([0]), torch.tensor([0.0]))  # max is not lowered
    assert buf.max_priority == pytest.approx(3.0, abs=1e-5)
    buf.add(obs=torch.tensor([9.0]), action=9)  # wraps onto slot 0
    assert buf.tree.priority(torch.tensor([0])).item() == pytest.approx(3.0, abs=1e-5)
    assert buf.tree.total == pytest.approx(6.0, abs=1e-4)


def test_per_zero_error_stays_sampleable_and_omega_flattens():
    buf = make_per(omega=0.5)
    buf.update_priorities(torch.arange(4), torch.tensor([0.0, 4.0, 4.0, 4.0]))
    assert buf.tree.priority(torch.tensor([0])).item() > 0
    assert buf.tree.priority(torch.tensor([1])).item() == pytest.approx(2.0, abs=1e-5)


def test_per_state_round_trip():
    buf = make_per()
    buf.update_priorities(torch.arange(4), torch.tensor([1.0, 2.0, 3.0, 4.0]))
    restored = make_per(n=0)
    restored.load_state_dict(buf.state_dict())
    assert restored.max_priority == buf.max_priority and restored.tree.total == buf.tree.total
    assert torch.equal(restored.tree.tree, buf.tree.tree)
    # a uniform run's buffer loads too: every transition starts at the same priority
    plain = ReplayBuffer(8, obs=(1,), action=())
    plain.add(obs=torch.tensor([1.0]), action=1)
    restored = make_per(n=0)
    restored.load_state_dict(plain.state_dict())
    assert restored.tree.total == 1.0 and restored.sample(4)["action"].tolist() == [1] * 4


SUPPORT = torch.linspace(-10, 10, 51)
UNIFORM = torch.full((1, 51), 1 / 51)


def test_projection_terminal_lands_on_reward():
    p = categorical_projection(UNIFORM, torch.tensor([1.3]), torch.tensor([0.0]), SUPPORT)
    assert p.sum().item() == pytest.approx(1.0, abs=1e-5)
    assert (p @ SUPPORT).item() == pytest.approx(1.3, abs=1e-4)
    assert (p > 0).sum() == 2  # split between the two neighbouring atoms


@pytest.mark.parametrize("ret,atom", [(2.0, 30), (-10.0, 0), (10.0, 50)])
def test_projection_keeps_all_mass_on_exact_atoms(ret, atom):
    p = categorical_projection(UNIFORM, torch.tensor([ret]), torch.tensor([0.0]), SUPPORT)
    assert p[0, atom].item() == pytest.approx(1.0, abs=1e-5)


def test_projection_preserves_mean_inside_support_and_saturates_outside():
    probs = torch.softmax(torch.randn(8, 51), -1)
    returns, discounts = torch.rand(8) * 0.5, torch.full((8,), 0.9)
    p = categorical_projection(probs, returns, discounts, SUPPORT)
    assert torch.allclose(p.sum(-1), torch.ones(8), atol=1e-5)
    assert torch.allclose(p @ SUPPORT, returns + discounts * (probs @ SUPPORT), atol=1e-3)
    p = categorical_projection(UNIFORM, torch.tensor([25.0]), torch.tensor([0.0]), SUPPORT)
    assert p[0, -1].item() == pytest.approx(1.0, abs=1e-5)


def test_noisy_linear_noise_only_in_train_mode():
    layer = NoisyLinear(4, 3)
    x = torch.ones(2, 4)
    first = layer(x)
    assert torch.equal(first, layer(x))  # fixed until the next reset
    layer.reset_noise()
    assert not torch.allclose(first, layer(x))
    layer.eval()
    mean = torch.nn.functional.linear(x, layer.weight, layer.bias)
    assert torch.equal(layer(x), mean)
    layer.reset_noise()
    assert torch.equal(layer(x), mean)
    assert "eps_in" not in layer.state_dict()  # noise is not checkpointed


@pytest.mark.parametrize("distributional", [True, False])
@pytest.mark.parametrize("dueling", [True, False])
@pytest.mark.parametrize("noisy", [True, False])
def test_qnetwork_shapes(distributional, dueling, noisy):
    net = QNetwork(distributional=distributional, dueling=dueling, noisy=noisy, atoms=11)
    x = torch.zeros(5, FEATURE_DIM)
    atoms = 11 if distributional else 1
    assert net.logits(x).shape == (5, NUM_ACTIONS, atoms)
    assert net(x).shape == (5, NUM_ACTIONS)
    assert net(x[0]).shape == (NUM_ACTIONS,)  # as QAgent calls it
    assert any(isinstance(m, NoisyLinear) for m in net.modules()) == noisy
    assert (net.value is not None) == dueling


def test_qnetwork_q_is_expectation_over_support():
    net = QNetwork(atoms=11, v_min=-1.0, v_max=1.0).eval()
    x = torch.randn(3, FEATURE_DIM)
    probs = net.logits(x).softmax(-1)
    assert torch.allclose(probs.sum(-1), torch.ones(3, NUM_ACTIONS), atol=1e-5)
    assert torch.allclose(net(x), (probs * net.support).sum(-1))
    assert net(x).abs().max() <= 1.0


def dqn_trainer(tmp_path, **overrides):
    settings = {**SMOKE["dqn"], **overrides}
    cfg = ALGORITHMS["dqn"].config_cls(
        mock=True,
        device="cpu",
        amp=False,
        total_steps=64,
        max_episode_steps=8,
        run_dir=str(tmp_path),
        run_name="rainbow",
        **settings,
    )
    envs = make_vec_env(mock=True, max_episode_steps=8)
    return ALGORITHMS["dqn"](cfg, envs), envs


def fill(trainer, steps=16):
    for _ in range(steps):
        trainer.train_iteration()


def test_rainbow_priorities_are_the_kl_loss(tmp_path, monkeypatch):
    trainer, envs = dqn_trainer(tmp_path, gradient_steps=1)
    fill(trainer, 16)
    seen = {}
    real = trainer.buffer.update_priorities

    def spy(idx, errors):
        seen["idx"], seen["errors"] = idx, errors
        real(idx, errors)

    monkeypatch.setattr(trainer.buffer, "update_priorities", spy)
    metrics = trainer._gradient_step()
    assert (seen["errors"] >= 0).all() and metrics["loss"] == pytest.approx(
        seen["errors"].mean().item(), abs=1e-6
    )
    expected = (seen["errors"].double() + trainer.buffer.PRIORITY_EPS) ** trainer.cfg.omega
    last = {int(i): p for i, p in zip(seen["idx"], expected, strict=True)}
    for i, p in last.items():
        assert trainer.buffer.tree.priority(torch.tensor([i])).item() == pytest.approx(p.item())
    assert 0 <= metrics["edge_mass"] <= 1
    envs.close()


def test_rainbow_double_q_online_selects_target_evaluates(tmp_path, monkeypatch):
    def counted(net, calls, key):
        real = net.forward

        def forward(x):
            calls[key] += 1
            return real(x)

        return forward

    for double, online_calls in ((True, 1), (False, 0)):
        trainer, envs = dqn_trainer(tmp_path, double=double, noisy=False)
        fill(trainer, 16)
        calls = {"online": 0, "target": 0}
        monkeypatch.setattr(trainer.q, "forward", counted(trainer.q, calls, "online"))
        monkeypatch.setattr(trainer.q_target, "forward", counted(trainer.q_target, calls, "target"))
        trainer._gradient_step()
        assert calls == {"online": online_calls, "target": 1 - online_calls}
        envs.close()


def test_rainbow_learns_a_fixed_transition(tmp_path):
    # a terminal transition with scaled reward 0.3: the expected value must converge to it
    trainer, envs = dqn_trainer(tmp_path, noisy=False, batch_size=8, lr=3e-3)
    obs = torch.zeros(FEATURE_DIM)
    trainer.buffer.size = 0
    for _ in range(8):
        trainer.buffer.add(obs=obs, action=2, reward=0.3, next_obs=obs, done=1.0, discount=0.9)
    for _ in range(300):
        metrics = trainer._gradient_step()
    q = trainer.q.eval()(obs)[2].item()
    assert q == pytest.approx(0.3, abs=0.05)
    assert metrics["q_mean"] == pytest.approx(q / trainer.cfg.reward_scale, rel=0.2)
    envs.close()


def test_rainbow_scales_rewards_and_syncs_target_periodically(tmp_path):
    trainer, envs = dqn_trainer(tmp_path, reward_scale=0.5, n_step=1, target_period=20)
    same = lambda: all(
        torch.equal(a, b) for a, b in zip(trainer.q.parameters(), trainer.q_target.parameters())
    )
    assert same()
    fill(trainer, 19)
    assert not same()  # updates have started, the target has not moved
    trainer.train_iteration()
    assert trainer.global_step == 20 and same()
    trainer.train_iteration()
    assert not same()
    # the mock's reward is the number of drills, 0 to 5
    assert trainer.buffer.data["reward"][: len(trainer.buffer)].max() <= 2.5
    assert (trainer.buffer.data["reward"][: len(trainer.buffer)] * 2 % 1 == 0).all()
    envs.close()


def test_rainbow_explores_with_noise_after_warmup(tmp_path):
    trainer, envs = dqn_trainer(tmp_path, learning_starts=16)
    assert trainer.epsilon() == 0.0
    obs = torch.zeros(1, FEATURE_DIM)
    trainer.global_step = 16
    # no epsilon: the variety of greedy actions comes from the noise alone
    assert len({trainer._select(obs)[0] for _ in range(200)}) > 1
    assert trainer.make_agent().net.training is False  # eval: mean weights, no noise
    actions = {trainer.make_agent().act(MockFactorioEnv().reset()[0]).name for _ in range(20)}
    assert len(actions) == 1
    trainer.train_iteration()  # training turns the noise back on
    assert trainer.q.training
    envs.close()
    trainer, envs = dqn_trainer(tmp_path, noisy=False, eps_end=0.05, eps_fraction=0.5)
    assert trainer.epsilon() == 1.0
    trainer.global_step = 32
    assert trainer.epsilon() == pytest.approx(0.05)
    envs.close()


def test_rainbow_beta_anneals_to_one(tmp_path):
    trainer, envs = dqn_trainer(tmp_path, beta_start=0.4)
    assert trainer.beta() == pytest.approx(0.4)
    trainer.global_step = 32
    assert trainer.beta() == pytest.approx(0.7)
    trainer.global_step = 640
    assert trainer.beta() == 1.0
    envs.close()


@pytest.mark.parametrize(
    "ablation",
    [
        {"double": False},
        {"prioritized": False},
        {"dueling": False},
        {"n_step": 1},
        {"distributional": False},
        {"noisy": False},
    ],
    ids=lambda a: next(iter(a)),
)
def test_rainbow_ablations_train_and_restore(tmp_path, ablation):
    trainer, envs = dqn_trainer(tmp_path, **ablation)
    path = trainer.train()
    restored = load_trainer(path, envs, device="cpu")
    assert len(restored.buffer) == len(trainer.buffer)
    assert load_agent(path).act(MockFactorioEnv(seed=0).reset()[0]) in MACRO_ACTIONS
    assert isinstance(trainer.buffer, PrioritizedReplayBuffer) == ablation.get("prioritized", True)
    metrics = trainer.train_iteration()
    assert ("edge_mass" in metrics) == ablation.get("distributional", True)
    envs.close()


def test_rainbow_resume_keeps_priorities_and_support(tmp_path):
    trainer, envs = dqn_trainer(tmp_path, v_min=-2.0, v_max=4.0)
    path = trainer.train()
    restored = load_trainer(path, envs, device="cpu")
    assert restored.buffer.max_priority == trainer.buffer.max_priority
    assert torch.equal(restored.buffer.tree.tree, trainer.buffer.tree.tree)
    assert restored.q.support[0] == -2.0 and restored.q.support[-1] == 4.0
    assert torch.equal(restored.q_target.encoder[0].weight, trainer.q_target.encoder[0].weight)
    envs.close()


def test_train_cli_exposes_rainbow_flags(tmp_path):
    rc = train_main(
        [
            "dqn",
            "--mock",
            "--total-steps",
            "24",
            "--learning-starts",
            "8",
            "--batch-size",
            "4",
            "--no-noisy",
            "--no-prioritized",
            "--v-min",
            "-2",
            "--device",
            "cpu",
            "--run-dir",
            str(tmp_path),
            "--run-name",
            "flags",
        ]
    )
    assert rc == 0
    ckpt = torch.load(tmp_path / "flags" / "checkpoint-final.pt", weights_only=False)
    assert ckpt["config"]["noisy"] is False and ckpt["config"]["v_min"] == -2.0


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
