# Typer CLI Integration for marimo Notebooks

## Why Typer

Typer provides a typed CLI layer that populates arguments before `app.run()` is called. This lets notebooks accept `--flags` when run as `uv run notebook.py`, while still showing the full interactive UI when opened with `marimo edit` or `marimo run`.

## How It Works

1. Declare a module-level `_cli_args: dict = {}` to hold parsed values
2. Create a `typer.Typer()` instance as `cli`
3. Define a `@cli.command()` that sets `_cli_args` values and calls `app.run()`
4. Replace `if __name__ == "__main__": app.run()` with `if __name__ == "__main__": cli()`
5. In cells, read from `_cli_args` when in script mode, fall back to widget values otherwise

## Skeleton Structure

```python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "typer",
# ]
# ///

import marimo
import typer  # module-level import for CLI setup only

__generated_with = "0.10.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import typer  # also import in setup cell for cell access

# Module-level CLI state
_cli_args: dict = {}
cli = typer.Typer()

@cli.command()
def run(
    # name: str = typer.Argument(help="Description of name"),
    # count: int = typer.Option(1, help="Number of iterations"),
    # verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
):
    # _cli_args["name"] = name
    # _cli_args["count"] = count
    # _cli_args["verbose"] = verbose
    app.run()

if __name__ == "__main__":
    cli()


@app.cell
def _(mo):
    is_script_mode = mo.app_meta().mode == "script"
    return (is_script_mode,)
```

## Complete Notebook Template

Ready-to-copy template with a positional `input_path` argument and a `count` option:

```python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "typer",
# ]
# ///

import marimo
import typer

from pathlib import Path

__generated_with = "0.10.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import typer
    from pathlib import Path

_cli_args: dict = {}
cli = typer.Typer()

@cli.command()
def run(
    input_path: Path = typer.Argument(help="Path to input file"),
    count: int = typer.Option(1, help="Number of iterations"),
):
    _cli_args["input_path"] = input_path
    _cli_args["count"] = count
    app.run()

if __name__ == "__main__":
    cli()


@app.cell
def _(mo):
    is_script_mode = mo.app_meta().mode == "script"
    return (is_script_mode,)


@app.cell
def _(mo, is_script_mode):
    path_input = mo.ui.text(placeholder="/path/to/file", label="Input file")
    path_input
    return (path_input,)


@app.cell
def _(is_script_mode, path_input):
    input_path = (
        _cli_args.get("input_path", Path("."))
        if is_script_mode
        else Path(path_input.value)
    )
    return (input_path,)


@app.cell
def _(mo, is_script_mode):
    count_slider = mo.ui.slider(start=1, stop=10, value=1, label="Count")
    count_slider
    return (count_slider,)


@app.cell
def _(is_script_mode, count_slider):
    count = _cli_args.get("count", 1) if is_script_mode else count_slider.value
    return (count,)
```

## Usage Examples

```bash
# Edit interactively (no CLI args)
uv run marimo edit notebook.py

# Run interactively in browser (no CLI args)
uv run marimo run notebook.py

# Run as script with positional arg and option
uv run notebook.py /path/to/data.csv --count 5

# Show help
uv run notebook.py --help

# Run with verbose flag
uv run notebook.py data.csv --verbose
```

## Argument Types Reference

| Type | Declaration | Example |
|---|---|---|
| Positional string | `name: str = typer.Argument(help="...")` | `uv run nb.py Alice` |
| Positional path | `path: Path = typer.Argument(help="...")` | `uv run nb.py data.csv` |
| Optional with default | `count: int = typer.Option(1, help="...")` | `--count 5` |
| Bool flag (true) | `verbose: bool = typer.Option(False, "--verbose", "-v")` | `--verbose` |
| Bool flag (false) | `debug: bool = typer.Option(True, "--no-debug")` | `--no-debug` |
| Enum/choice | `mode: str = typer.Option("fast", help="fast or slow")` | `--mode slow` |

## Rules

1. **`_cli_args` must be module-level** — declared outside any cell or function so all cells can read it
2. **`typer` imported in setup cell** — so cells can reference `typer` if needed (e.g., for type hints)
3. **`if __name__ == "__main__": cli()`** — replaces bare `app.run()`
4. **Always add `typer` to PEP 723 dependencies** when using CLI arguments
5. **Always show widgets** even in script mode — use `_cli_args` only to override the data source
6. **Prefer `_cli_args.get("key", default)`** over direct dict access to avoid KeyError in interactive mode
