import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    from dataclasses import asdict

    import marimo as mo
    import torch
    import typer

    from factorio_agent import MACRO_ACTIONS, featurize
    from factorio_agent import env as fenv
    from factorio_agent.agents import AGENTS, PolicyAgent


@app.cell
def _():
    mo.md("""
    # Factorio Learning Environment: untrained agent

    FLE actions are Python programs run in the game's REPL. `factorio_agent` wraps a set of
    **macro-actions** (code snippets) as a discrete action space, so a PyTorch policy can pick
    among them. Nothing here is trained yet.
    """)
    return


@app.cell
def _():
    is_script_mode = mo.app_meta().mode == "script"
    return (is_script_mode,)


@app.cell
def _():
    env_rows = [
        {"env_id": _id, "description": fenv.get_environment_info(_id)["description"]}
        for _id in fenv.list_available_environments()
    ]
    mo.vstack([mo.md(f"## Available tasks ({len(env_rows)})"), mo.ui.table(env_rows, page_size=8)])
    return


@app.cell
def _():
    mo.md("""
    ## Untrained policy on a synthetic starting observation

    Works without a Factorio server: featurize a lab-play style inventory and inspect the
    randomly-initialised policy's action distribution.
    """)
    return


@app.cell
def _():
    seed = mo.ui.number(start=0, stop=10_000, value=0, label="policy seed")
    seed
    return (seed,)


@app.cell
def _(seed):
    synthetic_obs = {
        "raw_text": "",
        "inventory": [
            {"type": "coal", "quantity": 500},
            {"type": "burner-mining-drill", "quantity": 50},
            {"type": "stone-furnace", "quantity": 10},
        ],
        "entities": [],
        "score": 0.0,
        "game_info": {"tick": 0, "time": 0.0, "speed": 1.0},
    }
    policy_agent = PolicyAgent(seed=seed.value)
    with torch.no_grad():
        _logits, _value = policy_agent.net(featurize(synthetic_obs))
    _probs = torch.softmax(_logits, dim=-1)
    action_probs = [
        {"action": _a.name, "prob": round(float(_p), 4)} for _a, _p in zip(MACRO_ACTIONS, _probs)
    ]
    mo.vstack(
        [
            mo.md(f"State value estimate: `{float(_value):.4f}`"),
            mo.ui.table(action_probs, selection=None),
        ]
    )
    return


@app.cell
def _():
    mo.md("""
    ## Play an episode

    Requires a running cluster (`uv run fle cluster start`).
    """)
    return


@app.cell
def _():
    try:
        containers = fenv.check_cluster()
        cluster_status = mo.md(f"Cluster up: `{', '.join(containers)}`").callout(kind="success")
    except RuntimeError as _e:
        containers = []
        cluster_status = mo.md(f"Cluster unavailable: {_e}").callout(kind="warn")
    cluster_status
    return (containers,)


@app.cell
def _():
    env_id = mo.ui.dropdown(
        options=fenv.list_available_environments(), value="iron_ore_throughput", label="task"
    )
    agent_name = mo.ui.dropdown(options=sorted(AGENTS), value="policy", label="agent")
    steps = mo.ui.slider(start=1, stop=50, value=8, label="steps", show_value=True)
    run_button = mo.ui.run_button(label="Run episode")
    mo.hstack([env_id, agent_name, steps, run_button], justify="start")
    return agent_name, env_id, run_button, steps


@app.cell
def _(agent_name, containers, env_id, is_script_mode, run_button, steps):
    mo.stop(not containers, mo.md("_Start the cluster to run an episode._"))
    mo.stop(not (run_button.value or is_script_mode), mo.md("_Click **Run episode**._"))
    _env = fenv.make_env(env_id.value)
    try:
        records = fenv.run_episode(_env, AGENTS[agent_name.value](), max_steps=steps.value)
    finally:
        _env.close()
    return (records,)


@app.cell
def _(records):
    _rows = [{k: v for k, v in asdict(_r).items() if k != "code"} for _r in records]
    mo.vstack(
        [
            mo.md(
                f"**{len(records)} steps**, total reward "
                f"`{sum(_r.reward for _r in records):.3f}`, "
                f"errors `{sum(_r.error for _r in records)}`"
            ),
            mo.ui.table(_rows, selection=None),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
