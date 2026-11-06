# MLX Function Transforms Reference

All in `mlx.core` (imported as `mx`).

## eval / async_eval

```python
mx.eval(*arrays)           # execute the lazy graph for these arrays (blocks until done)
mx.async_eval(*arrays)     # schedule evaluation without blocking Python

# Always eval at training step boundary:
mx.eval(loss, model.parameters())
```

## grad

```python
# Returns a function that computes the gradient wrt argnums (default: 0)
grad_fn = mx.grad(f, argnums=0)
# or by name:
grad_fn = mx.grad(f, argnames=["x"])

# Gradient of loss wrt first arg
grad_x = mx.grad(loss)(x, y)
```

**For models, prefer `nn.value_and_grad`** — it handles Module's non-array state correctly.

## value_and_grad

```python
# Returns (value, grad) in one pass
vg_fn = mx.value_and_grad(f, argnums=0)
loss, grad = vg_fn(x)

# For nn.Module:
vg_fn = nn.value_and_grad(model, loss_fn)
loss, grads = vg_fn(model, x, y)
```

## vmap

Vectorize a function over a batch dimension.

```python
batched_fn = mx.vmap(f, in_axes=(0, None), out_axes=0)
# in_axes: axis to map over for each input (None = broadcast, not mapped)
# out_axes: axis where batched results are stacked

# Example: per-sample gradients
per_sample_grad = mx.vmap(mx.grad(loss_per_sample))
grads = per_sample_grad(xs, ys)   # shape: (batch, ...)
```

## compile

Traces the computation graph on first call and caches for reuse. ~5x speedup for elementwise-heavy functions.

```python
compiled_fn = mx.compile(fn)

# With state (e.g., optimizer state or model params that change between calls)
from functools import partial
state = [model.state, optimizer.state]
@partial(mx.compile, inputs=state, outputs=state)
def train_step(x, y):
    loss, grads = vg_fn(model, x, y)
    optimizer.update(model, grads)
    return loss

# shapeless=True: don't recompile when input shapes change (unsafe if logic is shape-dependent)
compiled_fn = mx.compile(fn, shapeless=True)

# Disable/enable globally
mx.disable_compile()
mx.enable_compile()
```

**Compile rules:**
- Recompiles when input shapes, dtypes, or number of inputs change
- No side effects inside compiled fn (no eval/print during tracing)
- Compile the outermost function when composing transforms (e.g., compile(value_and_grad(f)))

## jvp / vjp

```python
# Jacobian-vector product (forward mode AD)
output, jvp = mx.jvp(f, primals, tangents)
# primals: list of input arrays
# tangents: list of arrays same shape as primals

# Vector-Jacobian product (reverse mode AD)
output, vjp_fn = mx.vjp(f, primals)
grads = vjp_fn(cotangents)
```

## checkpoint

Gradient checkpointing — trades memory for recompute.

```python
# Wrap a function to recompute activations during backward
checkpointed_fn = mx.checkpoint(f)

# Useful for large transformers:
class Block(nn.Module):
    def __call__(self, x):
        return mx.checkpoint(self._forward)(x)

    def _forward(self, x):
        return self.mlp(self.attn(x))
```

## custom_function

Define custom forward and backward passes.

```python
@mx.custom_function
def my_op(x, y):
    return x * y    # forward

@my_op.vjp
def my_op_vjp(primals, cotangent, output):
    x, y = primals
    return cotangent * y, cotangent * x    # grads wrt x and y
```

## Composing Transforms

Transforms compose — apply outermost last:

```python
# Per-sample gradients with vmap over grad
per_sample_grad_fn = mx.vmap(mx.grad(loss_per_sample))

# Compiled training step
train_step = mx.compile(mx.value_and_grad(loss_fn))

# Checkpoint inside grad
grad_fn = mx.grad(mx.checkpoint(f))
```

## Devices and Streams

```python
# Default stream is GPU
mx.default_stream(mx.gpu)
mx.default_stream(mx.cpu)

# Per-op override
out = mx.matmul(a, b, stream=mx.gpu)
out = mx.add(a, b, stream=mx.cpu)

# New stream for async parallelism
s = mx.new_stream(mx.gpu)
out = mx.matmul(a, b, stream=s)
```
