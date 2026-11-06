---
name: pytorch-guide-python
description: Write, debug, and optimize Python code using PyTorch 2.x. Covers tensors, nn.Module, training loops, optimizers, DataLoader, autograd, AMP, torch.compile, and torch.func transforms. Use when the user asks to "write pytorch code", "implement a model in pytorch", "train with pytorch", "use torch.nn", "add mixed precision", "compile a pytorch model", or "debug a pytorch training loop". Do NOT use for MLX, JAX, TensorFlow, or non-PyTorch frameworks.
argument-hint: "[task or description of what to implement]"
---

# PyTorch Python Guide

PyTorch 2.x reference for writing production-quality model code.

## Imports

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
```

## Device Setup

```python
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Move model and data to device
model = model.to(device)
x = x.to(device)
```

## Tensor Basics

```python
# Creation
x = torch.tensor([1.0, 2.0, 3.0])                        # from data
z = torch.zeros(3, 4)                                      # zeros
o = torch.ones(3, 4, dtype=torch.float16)
r = torch.arange(0, 10, step=2)
n = torch.randn(100, 10)                                   # standard normal
e = torch.empty(3, 4)                                      # uninitialized

# Attributes
x.shape   # torch.Size
x.dtype   # torch.float32 etc.
x.device  # device('cpu') or device('cuda:0')

# Move / cast
x = x.to(device)
x = x.float()   # cast to float32
x = x.half()    # float16
x = x.cuda()    # explicit GPU

# Shape manipulation
x.reshape(2, -1)
x.view(2, -1)        # zero-copy if contiguous
x.permute(2, 0, 1)
x.squeeze(0)
x.unsqueeze(0)
x.transpose(0, 1)
x.contiguous()       # make contiguous in memory

# Joining
torch.cat([a, b], dim=0)
torch.stack([a, b], dim=0)

# Detach from graph
x.detach()
x.detach_()   # in-place
```

## Defining a Model

```python
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)

model = MLP(784, 256, 10).to(device)
```

## Standard Training Loop

```python
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
loss_fn = nn.CrossEntropyLoss()

model.train()
for epoch in range(num_epochs):
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    scheduler.step()

# Evaluation
model.eval()
with torch.no_grad():
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        # compute metrics
```

## Mixed Precision (AMP) — default for training loops in this repo

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler(device.type, enabled=(device.type == "cuda"))

model.train()
for x, y in train_loader:
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()

    with autocast(device_type=device.type):   # float16 on CUDA, bfloat16 on CPU
        logits = model(x)
        loss = loss_fn(logits, y)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

Default to wrapping every PyTorch training loop in `autocast` + `GradScaler`
like this, unless the user explicitly asks for full-precision training. For
the full API (op-eligibility tables, gradient accumulation/penalty,
multi-model/DDP patterns, custom autograd `Function`s), see the
`pytorch-amp-guide-python` skill.

## Saving and Loading

```python
# Save checkpoint
torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epoch": epoch,
}, "checkpoint.pt")

# Load checkpoint
ckpt = torch.load("checkpoint.pt", map_location=device)
model.load_state_dict(ckpt["model"])
optimizer.load_state_dict(ckpt["optimizer"])
```

## torch.compile (2.x)

```python
model = torch.compile(model)                         # default: inductor backend
model = torch.compile(model, mode="reduce-overhead") # reduce Python overhead
model = torch.compile(model, mode="max-autotune")    # fullest optimization
model = torch.compile(model, fullgraph=True)         # error if graph breaks
model = torch.compile(model, dynamic=True)           # dynamic shapes
```

## Autograd

```python
x = torch.randn(3, requires_grad=True)
y = (x ** 2).sum()
y.backward()      # populates x.grad

# Disable gradient tracking
with torch.no_grad():
    val = model(x)

# Gradient of arbitrary fn
grads = torch.autograd.grad(outputs=y, inputs=x)
```

## References

- [TENSORS.md](references/TENSORS.md) — full tensor creation, math ops, indexing, linalg, fft, random
- [NN.md](references/NN.md) — all nn layers, activations, normalization, losses
- [OPTIMIZERS.md](references/OPTIMIZERS.md) — all optimizers and LR schedulers
- [TRAINING.md](references/TRAINING.md) — DataLoader, Dataset, AMP, DDP, profiler patterns
- [FUNC.md](references/FUNC.md) — torch.func (grad, vmap, jacrev), autograd.Function, distributions

## External Docs

- API reference: https://docs.pytorch.org/docs/2.12/pytorch-api.html
