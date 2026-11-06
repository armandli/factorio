# MLX Common Patterns

## Transformer / LLM Pattern

```python
class TransformerBlock(nn.Module):
    def __init__(self, dims, num_heads, mlp_dims):
        super().__init__()
        self.attn = nn.MultiHeadAttention(dims, num_heads)
        self.norm1 = nn.RMSNorm(dims)
        self.norm2 = nn.RMSNorm(dims)
        self.fc1 = nn.Linear(dims, mlp_dims)
        self.fc2 = nn.Linear(mlp_dims, dims)

    def __call__(self, x, mask=None):
        # Pre-norm transformer
        r = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), mask=mask)
        x = x + r
        r = self.fc2(nn.gelu(self.fc1(self.norm2(x))))
        return x + r

class Transformer(nn.Module):
    def __init__(self, vocab_size, dims, num_heads, mlp_dims, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dims)
        self.rope = nn.RoPE(dims // num_heads)
        self.layers = [TransformerBlock(dims, num_heads, mlp_dims) for _ in range(num_layers)]
        self.norm = nn.RMSNorm(dims)
        self.out_proj = nn.Linear(dims, vocab_size, bias=False)

    def __call__(self, x):
        x = self.embed(x)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.out_proj(self.norm(x))
```

## CNN Pattern (channels-last)

MLX Conv layers expect `(N, H, W, C)` — not `(N, C, H, W)` like PyTorch.

```python
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 8 * 8, 10)

    def __call__(self, x):
        # x: (N, H, W, C) — channels last
        x = nn.relu(self.conv1(x))
        x = self.pool(x)
        x = nn.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)
```

## Quantized Inference Pattern

```python
# Load a full-precision model then quantize weights
model = MyModel()
model.load_weights("model.npz")

# Quantize all Linear layers to 4-bit
def quantize_linear(module):
    if isinstance(module, nn.Linear):
        return nn.QuantizedLinear.from_linear(module, group_size=64, bits=4)
    return module

model.apply_to_modules(quantize_linear)
mx.eval(model.parameters())
```

## LoRA Fine-tuning Pattern

```python
class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank=8, alpha=16.0):
        super().__init__()
        self.linear = linear
        in_dim, out_dim = linear.weight.shape
        scale = alpha / rank
        self.lora_a = mx.random.normal((in_dim, rank)) * 0.01
        self.lora_b = mx.zeros((rank, out_dim))
        self.scale = scale

    def __call__(self, x):
        base = self.linear(x)
        lora = (x @ self.lora_a) @ self.lora_b
        return base + self.scale * lora

# Freeze base model, only train LoRA weights
model.freeze()
# Replace target layers with LoRA
model.attn.q_proj = LoRALinear(model.attn.q_proj, rank=8)
model.attn.v_proj = LoRALinear(model.attn.v_proj, rank=8)
model.unfreeze(keys=["lora_a", "lora_b"])
```

## Data Loading Pattern

MLX has no built-in DataLoader — use numpy/Python for batching, convert to mx.array per step:

```python
import numpy as np

def batch_iterate(batch_size, X, y, shuffle=True):
    n = X.shape[0]
    idx = np.random.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, batch_size):
        batch_idx = idx[i:i+batch_size]
        yield mx.array(X[batch_idx]), mx.array(y[batch_idx])

for x_batch, y_batch in batch_iterate(32, X_train, y_train):
    loss, grads = vg_fn(model, x_batch, y_batch)
    optimizer.update(model, grads)
    mx.eval(loss, model.parameters())
```

## Mixed Precision Pattern

```python
# Cast inputs to float16 at the boundary
x = x.astype(mx.float16)

# Keep weights in float32, cast inside forward
class MixedPrecisionLinear(nn.Linear):
    def __call__(self, x):
        return super().__call__(x.astype(mx.float16)).astype(mx.float32)
```

## Gradient Checkpointing for Large Models

```python
class CheckpointedBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def __call__(self, x):
        return mx.checkpoint(self.block)(x)

# Wrap every other block to halve activation memory
model.layers = [
    CheckpointedBlock(layer) if i % 2 == 0 else layer
    for i, layer in enumerate(model.layers)
]
```

## Distributed Training (multi-GPU)

```python
import mlx.core.distributed as dist

world = dist.init()
rank = world.rank()
size = world.size()

# All-reduce gradients across processes
def all_reduce_grads(grads):
    return mx.tree_map(lambda g: dist.all_reduce(g) / size, grads)

loss, grads = vg_fn(model, x, y)
grads = all_reduce_grads(grads)
optimizer.update(model, grads)
mx.eval(loss, model.parameters())
```

## Serialization Pattern

```python
# Save
weights = dict(mx.utils.tree_flatten(model.parameters()))
mx.save_safetensors("model.safetensors", weights)

# Load
weights = mx.load("model.safetensors")
model.update(mx.utils.tree_unflatten(list(weights.items())))
mx.eval(model.parameters())
```

## Common Pitfalls

| Issue | Cause | Fix |
|-------|-------|-----|
| Stale gradients | Missing `mx.eval` | Call `mx.eval(loss, model.parameters())` each step |
| Slow small ops | Running on GPU | Use `stream=mx.cpu` for elementwise/tiny ops |
| OOM on long sequence | No checkpointing | Wrap blocks with `mx.checkpoint` |
| Wrong conv output | PyTorch `(N,C,H,W)` habit | MLX uses `(N,H,W,C)` — transpose inputs |
| Recompile every step | Shape changes | Pad inputs to fixed shape or use `shapeless=True` |
| NaN gradients | No clipping | `optim.clip_grad_norm(grads, max_norm=1.0)` |
