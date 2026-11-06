---
name: mlx-guide-python
description: Write, debug, and optimize Python code using Apple's MLX framework for Apple Silicon. Covers arrays, neural network layers, optimizers, function transforms (grad/vmap/compile), lazy evaluation, and unified memory. Use when the user asks to "write mlx code", "implement a model in mlx", "use mlx for training", "convert pytorch to mlx", "add mlx grad/vmap/compile", or "build a neural network with mlx.nn". Do NOT use for PyTorch, JAX, or TensorFlow code without MLX, or for non-Apple hardware targets.
argument-hint: "[task or description of what to implement]"
---

# MLX Python Guide

MLX is Apple's array framework for Apple Silicon. Key differences from PyTorch/JAX:
- **Lazy evaluation** — operations build a graph; call `mx.eval()` to execute
- **Unified memory** — CPU and GPU share the same memory pool; no `.to(device)` needed
- **Composable transforms** — `grad`, `vmap`, `compile` compose like JAX

## Imports

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
```

## Array Basics

```python
# Creation
x = mx.array([1.0, 2.0, 3.0])
z = mx.zeros((3, 4))
o = mx.ones((3, 4), dtype=mx.float16)
r = mx.arange(0, 10, step=2)

# Ops are lazy — no computation until eval()
y = mx.matmul(x.reshape(1, -1), x.reshape(-1, 1))
mx.eval(y)   # triggers execution

# dtype casting
x = x.astype(mx.bfloat16)
```

## Lazy Evaluation — Critical Pattern

```python
# Correct: evaluate once per training step
loss, grads = value_and_grad_fn(model, batch)
optimizer.update(model, grads)
mx.eval(loss, model.parameters())   # single eval at step boundary

# Triggers implicit eval (avoid in hot paths):
print(x)          # implicit eval
x.item()          # implicit eval
x.tolist()        # implicit eval
```

## Defining a Model

```python
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def __call__(self, x):
        return self.fc2(nn.relu(self.fc1(x)))

model = MLP(784, 256, 10)
mx.eval(model.parameters())    # initialize weights
```

## Training Loop

```python
optimizer = optim.Adam(learning_rate=1e-3)

def loss_fn(model, x, y):
    logits = model(x)
    return nn.losses.cross_entropy(logits, y).mean()

value_and_grad_fn = nn.value_and_grad(model, loss_fn)

for batch_x, batch_y in dataloader:
    batch_x, batch_y = mx.array(batch_x), mx.array(batch_y)
    loss, grads = value_and_grad_fn(model, batch_x, batch_y)
    optimizer.update(model, grads)
    mx.eval(loss, model.parameters())
    print(f"loss: {loss.item():.4f}")
```

## Function Transforms

```python
# Gradient of a scalar function
grad_fn = mx.grad(loss_fn)

# Value and gradient together (preferred in training)
val_grad_fn = mx.value_and_grad(loss_fn)

# Vectorize over batch dimension
batched_fn = mx.vmap(single_sample_fn, in_axes=(0,), out_axes=0)

# Compile for speed (~5x on GELU, more for repeated shapes)
compiled_fn = mx.compile(fn)

# Compose: compile the outer transform
compiled_step = mx.compile(mx.value_and_grad(loss_fn))
```

## Freeze / Partial Training

```python
# Freeze all, then unfreeze specific layers
model.freeze()
model.fc2.unfreeze()

# Only frozen-excluded params get gradients
grads = mx.grad(loss_fn)(model.trainable_parameters(), x, y)
```

## Saving and Loading

```python
# Save
mx.save("weights.npz", dict(model.parameters()))
# or safetensors
mx.save_safetensors("weights.safetensors", dict(model.parameters()))

# Load
weights = mx.load("weights.npz")
model.update(weights)
mx.eval(model.parameters())
```

## Devices and Streams

```python
# Default: GPU. Override per-op with stream=
result = mx.matmul(a, b, stream=mx.gpu)     # GPU (default)
result = mx.add(a, b, stream=mx.cpu)        # CPU (better for small ops)
```

## Random Numbers

```python
mx.random.seed(42)
x = mx.random.normal(shape=(100, 10))
x = mx.random.uniform(low=0.0, high=1.0, shape=(100,))
key = mx.random.key(0)
sub_keys = mx.random.split(key, num=4)
```

## References

- [CORE-OPS.md](references/CORE-OPS.md) — full mlx.core operation catalogue
- [NN.md](references/NN.md) — all mlx.nn layers, activations, losses
- [OPTIMIZERS.md](references/OPTIMIZERS.md) — optimizers and LR schedulers
- [TRANSFORMS.md](references/TRANSFORMS.md) — grad, vmap, compile, jvp, vjp details
- [PATTERNS.md](references/PATTERNS.md) — common patterns: transformers, LSTMs, quantization, distributed

## External Docs

- API reference: https://ml-explore.github.io/mlx/build/html/index.html
- Examples repo: https://github.com/ml-explore/mlx-examples
