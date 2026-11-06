---
name: marimo-notebook
description: Write and edit marimo reactive Python notebooks in the correct file format with proper cell structure, setup cells, reactivity patterns, and Typer CLI argument support. Use when the user asks to "create a marimo notebook", "edit a marimo file", "add marimo cells", "make a marimo script", "add CLI arguments to a marimo notebook", or when editing a Python file that imports marimo. Do NOT use for general Python scripting without marimo, for Jupyter notebooks, or for explaining marimo concepts without making code changes.
argument-hint: "[notebook.py]"
---

# Notes for marimo Notebooks

## Running Marimo Notebooks

```bash
# Run as script (non-interactive, for testing)
uv run <notebook.py>

# Run interactively in browser
uv run marimo run <notebook.py>

# Edit interactively
uv run marimo edit <notebook.py>
```

## Required: Setup Cell

Every marimo notebook MUST have a `with app.setup:` block immediately after `app = marimo.App(...)`. Always import `marimo as mo` and `typer` in the setup cell:

```python
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import typer
```

Add any other shared imports (numpy, polars, torch, etc.) to the setup cell as well.

## Script Mode Detection

Use `mo.app_meta().mode == "script"` in a dedicated cell:

```python
@app.cell
def _(mo):
    is_script_mode = mo.app_meta().mode == "script"
    return (is_script_mode,)
```

## Key Principle: Show UI Always

**Always create and show widgets.** Only change the data source in script mode.

- Sliders, buttons, widgets should always be created and displayed
- In script mode, use synthetic/default data instead of waiting for user input
- Don't wrap everything in `if not is_script_mode` conditionals

### Good Pattern

```python
# Always show the widget
@app.cell
def _(mo):
    scatter_widget = mo.ui.anywidget(ScatterWidget())
    scatter_widget
    return (scatter_widget,)

# Only change data source based on mode
@app.cell
def _(is_script_mode, scatter_widget):
    if is_script_mode:
        X, y = make_moons(n_samples=200, noise=0.2)
    else:
        X, y = scatter_widget.widget.data_as_X_y
    return X, y

# Always show sliders - use .value in both modes
@app.cell
def _(mo):
    lr_slider = mo.ui.slider(start=0.001, stop=0.1, value=0.01)
    lr_slider
    return (lr_slider,)
```

## Don't Guard Cells with `if` Statements

Marimo's reactivity means cells only run when their dependencies are ready:

```python
# BAD - the if statement prevents the chart from showing
@app.cell
def _(plt, training_results):
    if training_results:  # WRONG
        fig, ax = plt.subplots()
        ax.plot(training_results['losses'])
        fig
    return

# GOOD - let marimo handle the dependency
@app.cell
def _(plt, training_results):
    fig, ax = plt.subplots()
    ax.plot(training_results['losses'])
    fig
    return
```

## Don't Use try/except for Control Flow

```python
# BAD - hiding errors behind try/except
@app.cell
def _(scatter_widget):
    try:
        X, y = scatter_widget.widget.data_as_X_y
    except Exception as e:
        return None, None

# GOOD - let it fail if something is wrong
@app.cell
def _(scatter_widget):
    X, y = scatter_widget.widget.data_as_X_y
    return X, y
```

Only use try/except for specific, expected exceptions with meaningful recovery.

## Cell Output Rendering

Marimo only renders the **final expression** of a cell:

```python
# BAD - indented expression won't render
@app.cell
def _(mo, condition):
    if condition:
        mo.md("This won't show!")  # WRONG - indented
    return

# GOOD - final expression renders
@app.cell
def _(mo, condition):
    result = mo.md("Shown!") if condition else mo.md("Also shown!")
    result
    return
```

## Marimo Variable Naming

Use underscore prefix for loop variables to make them cell-private:

```python
for _name, _model in items:
    ...
```

## PEP 723 Dependencies

```python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "typer",
#     "torch>=2.0.0",
# ]
# ///
```

Always add `typer` when using CLI arguments.

## Prefer pathlib over os.path

```python
# GOOD
from pathlib import Path
data_dir = Path(tempfile.mkdtemp())
parquet_file = data_dir / "data.parquet"

# BAD
import os
parquet_file = os.path.join(temp_dir, "data.parquet")
```

## marimo check

Run before delivering any notebook:

```bash
uvx marimo check <notebook.py>
```

## API Docs (Local)

```bash
uv --with marimo run python -c "import marimo as mo; help(mo.ui.form)"
```

## Typer CLI Arguments

Notebooks can accept `--flags` when run as `uv run notebook.py` using Typer. Use a module-level `_cli_args` dict to pass values from the CLI into cells. See [references/TYPER.md](references/TYPER.md) for the full integration pattern and template.

## Testing with pytest

Cells whose function names start with `test_` are discovered as pytest tests. Cell arguments inject other cells' return values as test inputs:

```bash
pytest notebook.py        # run all tests
pytest notebook.py -v     # verbose
```

See [references/PYTEST.md](references/PYTEST.md) for fixtures, parametrize, and class-based test patterns.

## Working with Polars

Import `polars as pl` in the setup cell. Marimo renders DataFrames natively — drop a DataFrame as the final cell expression, or use:
- `mo.ui.table(df)` — interactive table with row selection
- `mo.ui.dataframe(df)` — no-code filter/groupby GUI
- `mo.ui.data_explorer(df)` — column statistics and chart builder

See [references/POLARS.md](references/POLARS.md) for the full API.

## Charting with Altair

Import `altair as alt` in the setup cell. Wrap charts with `mo.ui.altair_chart(chart)` for reactivity — `.value` returns selected data as a DataFrame. See [references/ALTAIR.md](references/ALTAIR.md) for chart types and composition.

## NumPy Arrays

Import `numpy as np` in the setup cell. Use for numerical computation, array math, and linear algebra. See [references/NUMPY.md](references/NUMPY.md) for the full API.

## SciPy Scientific Computing

Import from `scipy` submodules in the setup cell (e.g., `from scipy import stats, optimize`). Use for statistics, optimization, signal processing, and integration. See [references/SCIPY.md](references/SCIPY.md) for the full API.

## PyTorch Neural Networks

Import `torch` and `torch.nn` in the setup cell. Define models with `nn.Module`, train with autograd. See [references/PYTORCH.md](references/PYTORCH.md) for the full API.

## Additional Resources

- [SQL.md](references/SQL.md) — SQL in marimo with DuckDB, SQLAlchemy, DuckDB connections
- [UI.md](references/UI.md) — marimo UI component reference
- [TOP-LEVEL-IMPORTS.md](references/TOP-LEVEL-IMPORTS.md) — exposing functions/classes as top-level imports
- [TYPER.md](references/TYPER.md) — Typer CLI argument integration
- [PYTEST.md](references/PYTEST.md) — pytest testing in marimo notebooks
- [POLARS.md](references/POLARS.md) — Polars DataFrame guide
- [ALTAIR.md](references/ALTAIR.md) — Altair charting guide
- [NUMPY.md](references/NUMPY.md) — NumPy array computing reference
- [SCIPY.md](references/SCIPY.md) — SciPy scientific computing reference
- [PYTORCH.md](references/PYTORCH.md) — PyTorch neural network reference
