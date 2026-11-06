# MLX Optimizers Reference

All in `mlx.optimizers` (imported as `optim`).

## Usage Pattern

```python
optimizer = optim.Adam(learning_rate=1e-3)

# In training loop:
loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
optimizer.update(model, grads)      # applies grads, updates optimizer state
mx.eval(loss, model.parameters())  # evaluate before next step
```

## Available Optimizers

### SGD
```python
optim.SGD(
    learning_rate,          # float or scheduler
    momentum=0.0,
    weight_decay=0.0,
    dampening=0.0,
    nesterov=False,
)
```

### Adam
```python
optim.Adam(
    learning_rate=1e-3,     # float or scheduler
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.0,
)
```

### AdamW
```python
optim.AdamW(
    learning_rate=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,      # decoupled weight decay (unlike Adam)
)
```

### Adamax
```python
optim.Adamax(
    learning_rate=2e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.0,
)
```

### Adafactor
```python
optim.Adafactor(
    learning_rate=None,     # None = use internal schedule
    eps=(1e-30, 1e-3),
    clip_threshold=1.0,
    decay_rate=-0.8,
    beta1=None,
    weight_decay=0.0,
    scale_parameter=True,
    relative_step=True,
    warmup_init=False,
)
```

### Adagrad
```python
optim.Adagrad(
    learning_rate=0.01,
    eps=1e-8,
    weight_decay=0.0,
)
```

### AdaDelta
```python
optim.AdaDelta(
    learning_rate=1.0,
    rho=0.9,
    eps=1e-6,
    weight_decay=0.0,
)
```

### RMSprop
```python
optim.RMSprop(
    learning_rate=0.01,
    alpha=0.99,
    eps=1e-8,
    weight_decay=0.0,
    momentum=0.0,
    centered=False,
)
```

### Lion
```python
optim.Lion(
    learning_rate=1e-4,
    betas=(0.9, 0.99),
    weight_decay=0.0,
)
```

### Muon
```python
optim.Muon(
    learning_rate=0.02,
    momentum=0.95,
    nesterov=True,
    ns_steps=6,             # Newton-Schulz iterations
    weight_decay=0.0,
)
```

### MultiOptimizer
```python
# Apply different optimizers to different parameter groups
optimizer = optim.MultiOptimizer(
    (optim.AdamW(lr=1e-3), ["fc1", "fc2"]),
    (optim.SGD(lr=1e-2),   ["classifier"]),
)
```

## Learning Rate Schedulers

Schedulers are callables that return the LR given the step count. Pass as `learning_rate=` to any optimizer.

```python
# Cosine decay: lr decays from init_lr to end_lr over decay_steps
schedule = optim.cosine_decay(init=1e-3, decay_steps=1000, end=1e-6)

# Exponential decay
schedule = optim.exponential_decay(init=1e-3, decay_rate=0.99)

# Linear schedule: interpolates from begin_value to end_value
schedule = optim.linear_schedule(init=0.0, end=1e-3, steps=100)

# Step decay: lr *= decay_rate every step_size steps
schedule = optim.step_decay(init=1e-3, decay_rate=0.5, step_size=100)

# Combine multiple schedules sequentially
schedule = optim.join_schedules(
    [optim.linear_schedule(0, 1e-3, 100), optim.cosine_decay(1e-3, 900)],
    boundaries=[100],
)

# Usage
optimizer = optim.AdamW(learning_rate=schedule)
```

## Gradient Clipping

```python
# Clip by global norm before update
grads = optim.clip_grad_norm(grads, max_norm=1.0)
```

## Saving/Loading Optimizer State

```python
# Save
state = optimizer.state
mx.save("optimizer_state.npz", state)

# Load
optimizer.state = mx.load("optimizer_state.npz")
mx.eval(optimizer.state)
```

## Full Training Loop Example

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

model = MyModel()
mx.eval(model.parameters())

schedule = optim.cosine_decay(init=3e-4, decay_steps=10_000)
optimizer = optim.AdamW(learning_rate=schedule, weight_decay=0.01)

def loss_fn(model, x, y):
    return nn.losses.cross_entropy(model(x), y).mean()

val_grad_fn = nn.value_and_grad(model, loss_fn)

model.train()
for step, (x, y) in enumerate(dataloader):
    x, y = mx.array(x), mx.array(y)
    loss, grads = val_grad_fn(model, x, y)
    grads = optim.clip_grad_norm(grads, max_norm=1.0)
    optimizer.update(model, grads)
    mx.eval(loss, model.parameters())
    if step % 100 == 0:
        print(f"step {step} loss {loss.item():.4f} lr {optimizer.learning_rate.item():.2e}")
```
