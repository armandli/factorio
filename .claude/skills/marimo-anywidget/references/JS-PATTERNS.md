# JavaScript Patterns for anywidget

Reference for common JavaScript patterns used in anywidget ESM front-ends.

---

## DOM Manipulation

### Basic element creation and cleanup

```javascript
function render({ model, el }) {
  const container = document.createElement("div");
  container.className = "my-widget";

  const label = document.createElement("span");
  label.textContent = model.get("label");

  container.appendChild(label);
  el.appendChild(container);

  const update = () => { label.textContent = model.get("label"); };
  model.on("change:label", update);

  return () => {
    model.off("change:label", update);
    // No need to remove el children — the framework handles that.
  };
}
```

### Building lists dynamically

```javascript
function render({ model, el }) {
  const ul = document.createElement("ul");
  el.appendChild(ul);

  const rebuild = () => {
    ul.innerHTML = "";  // clear
    for (const item of model.get("items")) {
      const li = document.createElement("li");
      li.textContent = item;
      ul.appendChild(li);
    }
  };
  rebuild();
  model.on("change:items", rebuild);

  return () => model.off("change:items", rebuild);
}
```

### Click handlers with state updates

```javascript
const btn = document.createElement("button");
btn.textContent = "Click me";

const onClick = () => {
  model.set("click_count", model.get("click_count") + 1);
  model.save_changes();
};
btn.addEventListener("click", onClick);

// Cleanup
return () => btn.removeEventListener("click", onClick);
```

### Input elements (text, range, checkbox)

```javascript
// Text input — debounce saves to avoid flooding Python
const input = document.createElement("input");
input.type = "text";
input.value = model.get("text");

let debounceTimer;
const onInput = () => {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    model.set("text", input.value);
    model.save_changes();
  }, 150);
};
input.addEventListener("input", onInput);

const syncToJS = () => { input.value = model.get("text"); };
model.on("change:text", syncToJS);

return () => {
  clearTimeout(debounceTimer);
  input.removeEventListener("input", onInput);
  model.off("change:text", syncToJS);
};
```

```javascript
// Range slider
const slider = document.createElement("input");
slider.type = "range";
slider.min = model.get("min_val");
slider.max = model.get("max_val");
slider.value = model.get("value");

const onChange = () => {
  model.set("value", Number(slider.value));
  model.save_changes();
};
slider.addEventListener("input", onChange);
model.on("change:value", () => { slider.value = model.get("value"); });
```

---

## State Synchronization Patterns

### Read / write cycle

```javascript
// Read
const val = model.get("trait_name");

// Write (stage + flush)
model.set("trait_name", newValue);
model.save_changes();

// Listen
const handler = () => { /* model.get("trait_name") has new value */ };
model.on("change:trait_name", handler);

// Listen to any change
model.on("change", () => {
  const changed = model.changed;  // object with keys that changed
});

// Remove listener
model.off("change:trait_name", handler);
```

### Two-way binding helper

```javascript
// Utility for keeping a DOM property in sync with a model trait
function bind(model, trait, getEl, setEl) {
  setEl(model.get(trait));
  const toJS = () => setEl(model.get(trait));
  const toPy = (val) => { model.set(trait, val); model.save_changes(); };
  model.on(`change:${trait}`, toJS);
  return { toPy, cleanup: () => model.off(`change:${trait}`, toJS) };
}
```

### Batch updates

```javascript
// Stage multiple changes before flushing
model.set("x", newX);
model.set("y", newY);
model.set("label", newLabel);
model.save_changes();  // single flush
```

---

## `initialize` vs `render`

| Hook | Runs | Use for |
|---|---|---|
| `initialize({ model })` | Once per model instance | Shared state, model-wide listeners, one-time setup |
| `render({ model, el })` | Once per view display | DOM construction, per-view listeners |

```javascript
function initialize({ model }) {
  // This data is shared across all views of this widget instance.
  model._shared = { history: [] };

  model.on("change:value", () => {
    model._shared.history.push(model.get("value"));
  });
}

function render({ model, el }) {
  // Each display gets its own DOM and listeners.
  const p = document.createElement("p");
  const update = () => {
    p.textContent = `value=${model.get("value")}, history length=${model._shared.history.length}`;
  };
  update();
  model.on("change:value", update);
  el.appendChild(p);
  return () => model.off("change:value", update);
}

export default { initialize, render };
```

Only use `initialize` when you genuinely need per-model (not per-view) state. Most widgets only need `render`.

---

## Traitlet Types and JS Equivalents

| Python traitlet | JS type | Notes |
|---|---|---|
| `traitlets.Int(0)` | `number` (integer) | Use `Math.floor` if rounding needed |
| `traitlets.Float(0.0)` | `number` | Full float precision |
| `traitlets.Unicode("")` | `string` | |
| `traitlets.Bool(False)` | `boolean` | |
| `traitlets.List([])` | `Array` | Elements must be JSON-serializable |
| `traitlets.Dict({})` | `object` | Values must be JSON-serializable |
| `traitlets.Tuple(())` | `Array` | Fixed-length, treated like List in JS |
| `traitlets.Bytes(b"")` | `DataView` | Binary; use for images, audio, raw buffers |

Nested structures (list of dicts, dict of lists) work fine as long as all leaf values are JSON-serializable. Update the entire object when mutating nested data — anywidget compares by reference:

```python
# Python — replace, don't mutate in-place
self.items = self.items + [new_item]  # triggers sync
# NOT: self.items.append(new_item)    # does NOT trigger sync
```

```javascript
// JS — replace array to trigger Python update
const items = [...model.get("items"), newItem];
model.set("items", items);
model.save_changes();
```

---

## Dark Mode CSS Patterns

Scope all selectors to a widget class. Use CSS custom properties for cleaner theming:

```css
.my-widget {
  --bg: #f5f5f5;
  --fg: #222;
  --border: #ccc;
  --accent: #4a90d9;
  --accent-hover: #357abd;
}

@media (prefers-color-scheme: dark) {
  .my-widget {
    --bg: #2a2a2a;
    --fg: #e0e0e0;
    --border: #555;
    --accent: #5ba3f0;
    --accent-hover: #4a90d9;
  }
}

.my-widget button {
  background: var(--bg);
  color: var(--fg);
  border: 1px solid var(--border);
}
.my-widget .primary {
  background: var(--accent);
  color: #fff;
}
.my-widget .primary:hover { background: var(--accent-hover); }
```

---

## Multi-View Widget Considerations

When the same widget instance is displayed in multiple cells, `render` runs once per display but `initialize` runs only once. Each view has its own `el`. Use `initialize` to set up shared model state and `render` to build isolated DOM per view.

```javascript
function initialize({ model }) {
  // Shared across all views
  model._views = new Set();
}

function render({ model, el }) {
  model._views.add(el);

  const p = document.createElement("p");
  el.appendChild(p);

  const update = () => { p.textContent = model.get("value"); };
  update();
  model.on("change:value", update);

  return () => {
    model.off("change:value", update);
    model._views.delete(el);
  };
}

export default { initialize, render };
```

---

## Animation and `requestAnimationFrame` Cleanup

Always cancel animation frames in the teardown function:

```javascript
function render({ model, el }) {
  const canvas = document.createElement("canvas");
  canvas.width = 400;
  canvas.height = 200;
  el.appendChild(canvas);
  const ctx = canvas.getContext("2d");

  let rafId;
  let running = true;

  const draw = () => {
    if (!running) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // ... draw frame using model.get(...)
    rafId = requestAnimationFrame(draw);
  };
  draw();

  model.on("change:running", () => {
    running = model.get("running");
    if (running) draw();
  });

  return () => {
    running = false;
    cancelAnimationFrame(rafId);
    model.off("change:running");
  };
}
```

For interval-based animation:

```javascript
const intervalId = setInterval(() => {
  model.set("tick", model.get("tick") + 1);
  model.save_changes();
}, 1000);

return () => clearInterval(intervalId);
```

---

## Canvas-Based Widgets

```javascript
function render({ model, el }) {
  const canvas = document.createElement("canvas");
  const width = model.get("width") || 400;
  const height = model.get("height") || 300;
  canvas.width = width;
  canvas.height = height;
  canvas.style.border = "1px solid var(--border, #ccc)";
  el.appendChild(canvas);

  const ctx = canvas.getContext("2d");

  const redraw = () => {
    ctx.clearRect(0, 0, width, height);
    const data = model.get("data");  // e.g., list of {x, y, r, color}
    for (const pt of data) {
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, pt.r, 0, 2 * Math.PI);
      ctx.fillStyle = pt.color || "#4a90d9";
      ctx.fill();
    }
  };
  redraw();
  model.on("change:data", redraw);

  // Mouse events — send click coordinates back to Python
  const onClick = (e) => {
    const rect = canvas.getBoundingClientRect();
    model.set("last_click", [e.clientX - rect.left, e.clientY - rect.top]);
    model.save_changes();
  };
  canvas.addEventListener("click", onClick);

  return () => {
    canvas.removeEventListener("click", onClick);
    model.off("change:data", redraw);
  };
}
```

For high-DPI displays:

```javascript
const dpr = window.devicePixelRatio || 1;
canvas.width = width * dpr;
canvas.height = height * dpr;
canvas.style.width = `${width}px`;
canvas.style.height = `${height}px`;
ctx.scale(dpr, dpr);
```

---

## SVG-Based Widgets (Vanilla, No D3)

```javascript
const SVG_NS = "http://www.w3.org/2000/svg";

function makeSvg(width, height) {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("width", width);
  svg.setAttribute("height", height);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  return svg;
}

function makeCircle(cx, cy, r, fill) {
  const c = document.createElementNS(SVG_NS, "circle");
  c.setAttribute("cx", cx);
  c.setAttribute("cy", cy);
  c.setAttribute("r", r);
  c.setAttribute("fill", fill);
  return c;
}

function render({ model, el }) {
  const width = 400, height = 300;
  const svg = makeSvg(width, height);
  el.appendChild(svg);

  const redraw = () => {
    svg.innerHTML = "";  // clear children
    for (const pt of model.get("points")) {
      svg.appendChild(makeCircle(pt.x, pt.y, pt.r ?? 5, pt.color ?? "#4a90d9"));
    }
  };
  redraw();
  model.on("change:points", redraw);

  // SVG click → send coordinates to Python
  const onClick = (e) => {
    const rect = svg.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * width;
    const y = ((e.clientY - rect.top) / rect.height) * height;
    model.set("last_click", [x, y]);
    model.save_changes();
  };
  svg.addEventListener("click", onClick);

  return () => {
    svg.removeEventListener("click", onClick);
    model.off("change:points", redraw);
  };
}

export default { render };
```

### Axis drawing (without D3)

```javascript
function drawAxis(svg, width, height, margin, xDomain, yDomain) {
  // X axis
  const xAxis = document.createElementNS(SVG_NS, "line");
  xAxis.setAttribute("x1", margin); xAxis.setAttribute("y1", height - margin);
  xAxis.setAttribute("x2", width - margin); xAxis.setAttribute("y2", height - margin);
  xAxis.setAttribute("stroke", "currentColor");
  svg.appendChild(xAxis);

  // Y axis
  const yAxis = document.createElementNS(SVG_NS, "line");
  yAxis.setAttribute("x1", margin); yAxis.setAttribute("y1", margin);
  yAxis.setAttribute("x2", margin); yAxis.setAttribute("y2", height - margin);
  yAxis.setAttribute("stroke", "currentColor");
  svg.appendChild(yAxis);
}

// Scale helpers
const xScale = (val, domain, range) =>
  range[0] + ((val - domain[0]) / (domain[1] - domain[0])) * (range[1] - range[0]);
```

---

## Summary Checklist

- [ ] `export default { render }` at the bottom (required)
- [ ] `render` returns a teardown function
- [ ] All `model.on(...)` calls have corresponding `model.off(...)` in teardown
- [ ] All event listeners have corresponding `removeEventListener` in teardown
- [ ] `cancelAnimationFrame` / `clearInterval` called in teardown if used
- [ ] `model.save_changes()` called after every `model.set()`
- [ ] CSS scoped to a widget-specific class
- [ ] Dark mode handled via `@media (prefers-color-scheme: dark)`
- [ ] Arrays/dicts replaced (not mutated) when triggering Python sync
