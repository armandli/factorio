# Altair Charting in marimo Notebooks

## Setup

```python
with app.setup:
    import marimo as mo
    import altair as alt
    import polars as pl  # or pandas as pd
```

Add `altair` to PEP 723 dependencies:

```python
# /// script
# dependencies = ["marimo", "altair", "polars"]
# ///
```

## Displaying Charts in marimo

- **`mo.ui.altair_chart(chart)`** — renders chart reactively; users can click/brush to select data; `.value` returns selected rows as a DataFrame
- **Bare `chart` as final expression** — also works but is non-interactive (no selection state)
- Prefer `mo.ui.altair_chart` for interactivity

```python
@app.cell
def _(df, alt, mo):
    chart = alt.Chart(df).mark_point().encode(x="x:Q", y="y:Q", color="category:N")
    reactive_chart = mo.ui.altair_chart(chart)
    reactive_chart
    return (reactive_chart,)

@app.cell
def _(reactive_chart):
    selected = reactive_chart.value  # polars/pandas DataFrame of selected points
    return (selected,)
```

## Core Chart Types

```python
# Scatter plot
alt.Chart(df).mark_point().encode(x="x:Q", y="y:Q")

# Line chart
alt.Chart(df).mark_line().encode(x="date:T", y="value:Q", color="series:N")

# Bar chart
alt.Chart(df).mark_bar().encode(x="category:N", y="count:Q", color="category:N")

# Area chart
alt.Chart(df).mark_area(opacity=0.4).encode(x="x:Q", y="y:Q")

# Histogram (binned bar)
alt.Chart(df).mark_bar().encode(alt.X("value:Q", bin=True), y="count()")

# Heatmap
alt.Chart(df).mark_rect().encode(x="x:O", y="y:O", color="value:Q")
```

## Encoding Type Shorthand

| Shorthand | Meaning | Example |
|---|---|---|
| `:Q` | Quantitative (numeric) | `"price:Q"` |
| `:N` | Nominal (categorical) | `"category:N"` |
| `:O` | Ordinal (ordered categories) | `"rank:O"` |
| `:T` | Temporal (date/time) | `"date:T"` |

## Common Encodings

```python
.encode(
    x="col:Q",
    y="col:Q",
    color="col:N",
    size="col:Q",
    shape="col:N",
    tooltip=["col1", "col2", "col3"],
    opacity=alt.value(0.7),
)
```

## Chart Properties

```python
.properties(width=600, height=400, title="My Chart")
```

## Transforms

```python
# Filter
.transform_filter(alt.datum.value > 0)

# Aggregate
.transform_aggregate(mean_val="mean(value)", groupby=["category"])

# Calculate derived field
.transform_calculate(log_val="log(datum.value)")

# Fold (wide to long)
.transform_fold(["col_a", "col_b"], as_=["variable", "value"])
```

## Composition

```python
# Layer: overlay two charts
line + points

# Horizontal: side by side
chart1 | chart2

# Vertical: stacked
chart1 & chart2

# Facet: small multiples
alt.Chart(df).mark_point().encode(
    x="x:Q", y="y:Q"
).facet(column="category:N")
```

## Interactive Selections

```python
brush = alt.selection_interval()
color = alt.condition(brush, "category:N", alt.value("lightgray"))
chart = alt.Chart(df).mark_point().encode(x="x:Q", y="y:Q", color=color).add_params(brush)
```

For marimo interactivity, prefer `mo.ui.altair_chart` over manual `add_params` — marimo handles selection state reactively.

## Axis and Legend Customization

```python
.encode(
    x=alt.X("value:Q", axis=alt.Axis(title="Custom Title", format=".2f")),
    color=alt.Color("cat:N", legend=alt.Legend(orient="bottom")),
)
```
