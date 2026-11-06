---
name: marimo-anywidget
description: Create custom interactive widgets using anywidget in marimo notebooks, combining Python traitlets state with vanilla JavaScript ESM front-ends. Use when the user asks to "create a custom widget in marimo", "add a JavaScript widget", "build an interactive marimo widget", "inject JavaScript into marimo", or "write an anywidget". Do NOT use for general marimo notebook editing (use marimo-notebook), for using existing anywidget-based libraries without customization, or for non-interactive display-only elements.
argument-hint: "[widget-name] [description of widget behavior]"
---

# marimo anywidget

anywidget lets you build custom interactive widgets that sync Python state to a JavaScript front-end. The three parts are:

- **Python**: `anywidget.AnyWidget` subclass with `traitlets` for synced state
- **JavaScript**: ESM module with `render` (and optionally `initialize`) functions
- **Marimo**: `mo.ui.anywidget()` wrapper that integrates the widget into marimo's reactive graph

## Python Widget Class

```python
import anywidget
import traitlets

class MyWidget(anywidget.AnyWidget):
    _esm = """..."""    # required: JS module string or Path
    _css = """..."""    # optional: CSS string or Path

    # Synced traits — changes propagate to JS and back
    value    = traitlets.Int(0).tag(sync=True)
    label    = traitlets.Unicode("hello").tag(sync=True)
    items    = traitlets.List([]).tag(sync=True)
    options  = traitlets.Dict({}).tag(sync=True)
    enabled  = traitlets.Bool(True).tag(sync=True)
    ratio    = traitlets.Float(0.5).tag(sync=True)
```

Only traits tagged with `sync=True` cross the Python–JS boundary. Private traits (no `sync=True`) stay Python-only.

**Common traitlet types:**

| Traitlet | JS type | Notes |
|---|---|---|
| `Int` | `number` (integer) | |
| `Float` | `number` | |
| `Unicode` | `string` | |
| `Bool` | `boolean` | |
| `List` | `Array` | elements must be JSON-serializable |
| `Dict` | `object` | values must be JSON-serializable |
| `Bytes` | `DataView` | for binary data |

## JavaScript Front-End (AFM)

The JS module must export a default object with a `render` function. Optionally include `initialize`.

```javascript
function initialize({ model }) {
  // Runs ONCE per model instance (even if displayed multiple times).
  // Good for one-time setup: allocating shared state, attaching model-wide listeners.
}

function render({ model, el }) {
  // Runs ONCE per view (each time the widget is displayed).
  // Build your DOM here and attach listeners.
  // Must return a cleanup function.

  const btn = document.createElement("button");
  btn.textContent = `count: ${model.get("value")}`;

  btn.addEventListener("click", () => {
    model.set("value", model.get("value") + 1);
    model.save_changes();  // flush to Python
  });

  const update = () => {
    btn.textContent = `count: ${model.get("value")}`;
  };
  model.on("change:value", update);

  el.appendChild(btn);

  // Return cleanup — called when the view is removed
  return () => {
    model.off("change:value", update);
  };
}

export default { initialize, render };
```

**Model API:**

| Method | Purpose |
|---|---|
| `model.get("trait")` | Read current value |
| `model.set("trait", val)` | Stage a change |
| `model.save_changes()` | Flush staged changes to Python |
| `model.on("change:trait", cb)` | Listen for change |
| `model.off("change:trait", cb)` | Remove listener |
| `model.on("change", cb)` | Listen for any change |

Always call `model.save_changes()` after `model.set()`. Always clean up listeners in the returned teardown function.

## CSS Styling

Keep CSS minimal. Use `@media (prefers-color-scheme: dark)` for dark mode:

```css
.my-widget button {
  font-size: 14px;
  padding: 6px 12px;
  border: 1px solid #ccc;
  border-radius: 4px;
  background: #f5f5f5;
  cursor: pointer;
}
.my-widget button:hover { background: #e0e0e0; }

@media (prefers-color-scheme: dark) {
  .my-widget button {
    background: #2a2a2a;
    border-color: #555;
    color: #eee;
  }
  .my-widget button:hover { background: #3a3a3a; }
}
```

Scope all selectors to a widget-specific class to avoid leaking into the page.

## Marimo Integration

```python
import marimo as mo

widget = mo.ui.anywidget(MyWidget())
widget          # display — return as final cell expression
widget.value    # dict of all synced traits: {"value": 0, "label": "hello", ...}
widget.value["value"]   # access specific trait
widget.widget   # underlying AnyWidget instance (for calling Python methods)
```

Use `widget.value` as a reactive dependency in downstream cells — marimo re-runs those cells whenever any synced trait changes.

```python
@app.cell
def _(widget):
    count = widget.value["value"]
    return (count,)

@app.cell
def _(count, mo):
    mo.md(f"Current count: **{count}**")
    return
```

## External Files

Use `pathlib.Path` when the JS or CSS is large enough that embedding it in the Python string becomes unwieldy:

```python
from pathlib import Path

class MyWidget(anywidget.AnyWidget):
    _esm = Path(__file__).parent / "widget.js"
    _css = Path(__file__).parent / "widget.css"
    value = traitlets.Int(0).tag(sync=True)
```

anywidget watches the files for changes in development mode, giving live-reload behavior.

## Worked Example — Counter Widget

A complete counter widget showing the full Python + JS + CSS + marimo pattern with proper cleanup:

```python
import anywidget
import traitlets
import marimo as mo

class CounterWidget(anywidget.AnyWidget):
    _esm = """
    function render({ model, el }) {
      // Build DOM
      const container = document.createElement("div");
      container.className = "counter-widget";

      const display = document.createElement("span");
      display.className = "counter-value";

      const dec = document.createElement("button");
      dec.textContent = "−";

      const inc = document.createElement("button");
      inc.textContent = "+";

      const reset = document.createElement("button");
      reset.className = "counter-reset";
      reset.textContent = "reset";

      container.append(dec, display, inc, reset);
      el.appendChild(container);

      // Sync helpers
      const refresh = () => {
        display.textContent = model.get("count");
        dec.disabled = model.get("count") <= model.get("min_val");
        inc.disabled = model.get("count") >= model.get("max_val");
      };
      refresh();

      // Handlers
      const onInc = () => {
        model.set("count", Math.min(model.get("count") + model.get("step"), model.get("max_val")));
        model.save_changes();
      };
      const onDec = () => {
        model.set("count", Math.max(model.get("count") - model.get("step"), model.get("min_val")));
        model.save_changes();
      };
      const onReset = () => {
        model.set("count", model.get("initial"));
        model.save_changes();
      };

      inc.addEventListener("click", onInc);
      dec.addEventListener("click", onDec);
      reset.addEventListener("click", onReset);
      model.on("change:count", refresh);

      // Cleanup
      return () => {
        inc.removeEventListener("click", onInc);
        dec.removeEventListener("click", onDec);
        reset.removeEventListener("click", onReset);
        model.off("change:count", refresh);
      };
    }

    export default { render };
    """

    _css = """
    .counter-widget {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-family: monospace;
    }
    .counter-widget button {
      width: 32px;
      height: 32px;
      border: 1px solid #bbb;
      border-radius: 4px;
      background: #f0f0f0;
      cursor: pointer;
      font-size: 18px;
      line-height: 1;
    }
    .counter-widget button:disabled { opacity: 0.4; cursor: default; }
    .counter-widget button:not(:disabled):hover { background: #ddd; }
    .counter-reset {
      font-size: 12px !important;
      width: auto !important;
      padding: 0 8px;
    }
    .counter-value {
      min-width: 3ch;
      text-align: center;
      font-size: 20px;
    }
    @media (prefers-color-scheme: dark) {
      .counter-widget button {
        background: #2e2e2e;
        border-color: #555;
        color: #e0e0e0;
      }
      .counter-widget button:not(:disabled):hover { background: #3e3e3e; }
    }
    """

    count   = traitlets.Int(0).tag(sync=True)
    initial = traitlets.Int(0).tag(sync=True)
    step    = traitlets.Int(1).tag(sync=True)
    min_val = traitlets.Int(0).tag(sync=True)
    max_val = traitlets.Int(10).tag(sync=True)


# Display
app = mo.App()

with app.setup:
    import marimo as mo
    import anywidget
    import traitlets

@app.cell
def _():
    counter = mo.ui.anywidget(CounterWidget(min_val=0, max_val=20, step=2))
    counter
    return (counter,)

@app.cell
def _(counter, mo):
    mo.md(f"Count is **{counter.value['count']}**")
    return
```

## Best Practices

- **Always export default** `{ render }` or `{ initialize, render }` — the widget will silently fail without it.
- **Always clean up** listeners and animations in the teardown function returned from `render`.
- **Call `model.save_changes()`** after every `model.set()` or Python won't see the update.
- **Scope CSS selectors** to a widget-specific class so styles don't leak to the page.
- **Use `initialize` only** when you need truly global (per-model) setup; `render` runs per view.
- **Prefer external files** (`pathlib.Path`) when JS or CSS exceeds ~50 lines — easier to edit and gets live-reload.
- **Keep synced state minimal** — large objects (big arrays, images) hurt performance; serialize only what the UI actually needs.
- **Validate in Python**, not JS — raise `traitlets.TraitError` on bad input in `__init__` or `observe` handlers.

See [references/JS-PATTERNS.md](references/JS-PATTERNS.md) for DOM manipulation patterns, animation cleanup, canvas, SVG, and multi-view considerations.
