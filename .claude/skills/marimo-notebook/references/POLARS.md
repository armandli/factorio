# Polars DataFrames in marimo Notebooks

## Setup

```python
with app.setup:
    import marimo as mo
    import polars as pl
```

Add `polars` to PEP 723 dependencies:

```python
# /// script
# dependencies = ["marimo", "polars"]
# ///
```

## Displaying DataFrames in marimo

- **Final expression**: Drop a DataFrame as the last line of a cell — marimo renders it natively with paging, sorting, and filtering
- **`mo.ui.table(df)`** — interactive table with row selection; `table.value` returns selected rows as a DataFrame
- **`mo.ui.dataframe(df)`** — no-code GUI for filters, groupby, aggregation; `transformed.value` returns result
- **`mo.ui.data_explorer(df)`** — column-level statistics and a chart builder
- **`mo.plain(df)`** — disable rich viewer and show raw repr

## Reading Data

```python
df = pl.read_csv("data.csv")
df = pl.read_parquet("data.parquet")
df = pl.read_json("data.json")
df = pl.read_ndjson("data.ndjson")
```

## Core DataFrame Operations

```python
# Select columns
df.select(["col1", "col2"])
df.select(pl.col("name"), pl.col("age"))

# Filter rows
df.filter(pl.col("age") > 30)
df.filter((pl.col("age") > 20) & (pl.col("active") == True))

# Add/modify columns
df.with_columns(
    (pl.col("price") * 1.1).alias("adjusted_price"),
    pl.col("name").str.to_uppercase().alias("name_upper"),
)

# Sort
df.sort("age", descending=True)

# Group by and aggregate
df.group_by("region").agg(
    pl.col("sales").sum().alias("total"),
    pl.col("sales").mean().alias("avg"),
    pl.len().alias("count"),
)

# Join
df1.join(df2, on="id", how="inner")  # inner, left, right, full, semi, anti
```

## LazyFrame (query optimization)

Use `lazy()` for large datasets — builds a query plan and executes only when `.collect()` is called:

```python
result = (
    pl.scan_csv("large.csv")
    .filter(pl.col("amount") > 100)
    .group_by("category")
    .agg(pl.col("amount").sum())
    .collect()  # execute the query plan
)
```

Prefer LazyFrame for pipelines on large data; use `pl.scan_*` readers instead of `pl.read_*`.

## Expressions

```python
pl.col("name")                              # column reference
pl.lit(42)                                  # literal value
pl.col("price") * 1.1                       # arithmetic
pl.col("name").str.contains("foo")          # string ops
pl.col("date").dt.year()                    # datetime ops
pl.col("tags").list.len()                   # list ops
pl.when(pl.col("x") > 0).then(pl.lit("pos")).otherwise(pl.lit("neg"))  # conditional
```

## Writing Data

```python
df.write_csv("output.csv")
df.write_parquet("output.parquet")
df.write_json("output.json")
```

## Integration with mo.sql

marimo's `mo.sql()` uses DuckDB, which can query Polars DataFrames directly by variable name:

```python
@app.cell
def _(df, mo):
    result = mo.sql("SELECT category, SUM(amount) FROM df GROUP BY category")
    result  # returns polars DataFrame
    return (result,)
```

## Reactive Filtering Pattern

```python
@app.cell
def _(mo):
    age_filter = mo.ui.slider(start=0, stop=100, value=50, label="Max age")
    age_filter
    return (age_filter,)

@app.cell
def _(df, age_filter, mo):
    filtered = df.filter(pl.col("age") < age_filter.value)
    mo.ui.table(filtered)
    return (filtered,)
```
