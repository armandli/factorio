# pytest Testing in marimo Notebooks

## Overview

marimo notebooks are Python files — run `pytest notebook.py` directly without any plugins or configuration. Pytest discovers test cells automatically.

## Test Cell Pattern

Cells whose **function names start with `test_`** are discovered as pytest tests. The cell's arguments inject other cells' return values as test inputs:

```python
@app.cell
def _():
    def inc(x):
        return x + 1
    return (inc,)

@app.cell
def test_sanity(inc):      # `inc` comes from the cell above
    assert inc(3) == 4
```

The cell argument `inc` is automatically resolved from the cell that returns `(inc,)`.

## Critical Rule

A test cell must contain **only** test functions, fixtures, or test classes. Any mix with helpers, imports, or constants causes the cell to be skipped by pytest.

```python
# BAD - mixing test with helper causes cell to be skipped
@app.cell
def test_stuff(inc):
    CONSTANT = 42           # WRONG - non-test code
    def helper(): pass      # WRONG - non-test code
    def test_inc(inc):
        assert inc(1) == 2

# GOOD - test cell contains only test code
@app.cell
def test_inc(inc):
    assert inc(3) == 4
```

## Running Tests

```bash
# Run all tests in notebook
pytest notebook.py

# Verbose output
pytest notebook.py -v

# Filter by name
pytest -k test_sanity

# Run with coverage
pytest notebook.py --cov
```

## Fixtures

### Setup Cell Fixtures (shared across all test cells)

Fixtures defined in the `with app.setup:` block are available to all test cells:

```python
with app.setup:
    import marimo as mo
    import pytest

    @pytest.fixture
    def sample_data():
        return [1, 2, 3, 4, 5]
```

### In-Cell Fixtures (local to that cell only)

Fixtures defined in a regular cell are available only within that same cell — they cannot be shared with other cells:

```python
@app.cell
def _(pytest):
    @pytest.fixture
    def temp_file(tmp_path):
        return tmp_path / "test.txt"

    def test_file_created(temp_file):
        temp_file.write_text("hello")
        assert temp_file.read_text() == "hello"
```

### conftest.py Fixtures

Standard `conftest.py` fixtures work normally and are available to all test cells.

## Class-Based Tests

```python
@app.cell
def _(pytest):
    class TestMyFeature:
        @pytest.fixture(scope="class")
        def connection(self):
            return create_connection()

        def test_query(self, connection):
            assert connection.query("SELECT 1") == 1

        def test_empty(self, connection):
            assert connection.query("SELECT 0") == 0
```

## Parametrize

```python
@app.cell
def _(pytest, inc):
    @pytest.mark.parametrize("x,expected", [(1, 2), (3, 4), (9, 10)])
    def test_inc(x, expected):
        assert inc(x) == expected
```

## Adding pytest to Dependencies

For notebooks with tests, add `pytest` to the PEP 723 block:

```python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "pytest",
# ]
# ///
```

## Key Takeaways

- **Cell function names starting with `test_`** are auto-discovered by pytest
- **Test cells should contain only test code** — keep helpers in separate cells
- **Fixtures must be in the setup cell or `conftest.py`** to be shared across test cells
- **Cell arguments** automatically inject return values from other cells as test dependencies
- **No pytest plugin required** — marimo notebooks work with stock pytest
