import json

from factorio_agent.rl.compare import _step_auc, compare


def _write_run(root, name, algo, rows):
    run = root / name
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"algo": algo, "total_steps": 1000}))
    (run / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return run


def test_step_auc_holds_values_until_next_point():
    assert _step_auc([(0, 1.0), (50, 3.0)], 100) == 2.0
    assert _step_auc([(0, 1.0), (200, 9.0)], 100) == 1.0  # points past the budget are ignored
    assert _step_auc([], 100) is None


def test_compare_cuts_runs_at_common_budget_and_ranks_by_eval_auc(tmp_path):
    # "fast" learns early; "late" ends higher but only after the common budget of 100 steps
    fast = [{"global_step": s, "episode_return": 1.0} for s in (20, 40, 60, 80, 100)]
    fast += [
        {"global_step": 100, "eval_step": s, "eval_return": v} for s, v in ((0, 0.0), (50, 5.0))
    ]
    late = [{"global_step": s, "episode_return": 1.0} for s in (50, 100, 150, 200)]
    late += [
        {"global_step": 200, "eval_step": s, "eval_return": v}
        for s, v in ((0, 0.0), (50, 1.0), (150, 50.0))
    ]
    a = _write_run(tmp_path, "fast", "ppo", fast)
    b = _write_run(tmp_path, "late", "dqn", late)
    result = compare([a, b])
    assert result["budget"] == 100
    assert result["decided_by"] == "eval_auc"
    assert result["winner"] == str(a)
    late_row = next(r for r in result["runs"] if r["run"] == str(b))
    assert late_row["best_eval"] == 1.0  # the 50.0 eval at step 150 is outside the budget


def test_compare_falls_back_to_train_auc_without_enough_evals(tmp_path):
    a = _write_run(
        tmp_path, "a", "ppo", [{"global_step": s, "episode_return": 1.0} for s in (10, 20)]
    )
    b = _write_run(
        tmp_path, "b", "dqn", [{"global_step": s, "episode_return": 2.0} for s in (10, 20)]
    )
    result = compare([a, b])
    assert result["decided_by"] == "train_auc"
    assert result["winner"] == str(b)


def test_compare_metric_train_overrides_eval(tmp_path):
    a = [{"global_step": s, "episode_return": 1.0} for s in (10, 20, 30)]
    a += [{"global_step": 30, "eval_step": s, "eval_return": 9.0} for s in (10, 20)]
    b = [{"global_step": s, "episode_return": 5.0} for s in (10, 20, 30)]
    b += [{"global_step": 30, "eval_step": s, "eval_return": 0.0} for s in (10, 20)]
    ra, rb = _write_run(tmp_path, "a", "ppo", a), _write_run(tmp_path, "b", "mcgs", b)
    assert compare([ra, rb])["winner"] == str(ra)
    result = compare([ra, rb], metric="train")
    assert result["decided_by"] == "train_auc"
    assert result["winner"] == str(rb)
