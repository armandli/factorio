# PyTorch Optimizers and LR Schedulers Reference

All in `torch.optim`.

## Optimizer Usage Pattern

```python
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# Training step
optimizer.zero_grad(set_to_none=True)   # set_to_none=True is faster than filling with 0
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()

# Save / load state
torch.save(optimizer.state_dict(), "optimizer.pt")
optimizer.load_state_dict(torch.load("optimizer.pt"))
```

## Available Optimizers

### SGD
```python
optim.SGD(params, lr, momentum=0.0, dampening=0.0,
          weight_decay=0.0, nesterov=False)
# Example
optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
```

### Adam
```python
optim.Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
           weight_decay=0.0, amsgrad=False)
```

### AdamW (preferred over Adam for transformers)
```python
optim.AdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=0.01, amsgrad=False)
```

### NAdam
```python
optim.NAdam(params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=0.0, momentum_decay=4e-3)
```

### RAdam
```python
optim.RAdam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
```

### Adamax
```python
optim.Adamax(params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
```

### Adagrad
```python
optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0, eps=1e-10)
```

### Adafactor
```python
optim.Adafactor(params, lr=None, eps=(1e-30, 1e-3), clip_threshold=1.0,
                decay_rate=-0.8, beta1=None, weight_decay=0.0,
                scale_parameter=True, relative_step=True, warmup_init=False)
```

### Adadelta
```python
optim.Adadelta(params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0.0)
```

### RMSprop
```python
optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0.0,
              momentum=0.0, centered=False)
```

### Rprop
```python
optim.Rprop(params, lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-6, 50))
```

### ASGD
```python
optim.ASGD(params, lr=0.01, lambd=1e-4, alpha=0.75, t0=1e6, weight_decay=0.0)
```

### LBFGS (for small models / full-batch)
```python
optim.LBFGS(params, lr=1, max_iter=20, max_eval=None,
            tolerance_grad=1e-7, tolerance_change=1e-9,
            history_size=100, line_search_fn=None)
# Requires closure:
def closure():
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    return loss
optimizer.step(closure)
```

### Muon
```python
optim.Muon(params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=6, weight_decay=0.0)
```

### SparseAdam
```python
optim.SparseAdam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
# For sparse gradients (embeddings)
```

## Parameter Groups

Different hyperparameters for different parts of the model:

```python
optimizer = optim.AdamW([
    {"params": model.encoder.parameters(), "lr": 1e-4},
    {"params": model.head.parameters(),    "lr": 1e-3, "weight_decay": 0.0},
], weight_decay=0.01)
```

## LR Schedulers

All in `torch.optim.lr_scheduler`. Call `scheduler.step()` after each epoch (or step).

```python
from torch.optim import lr_scheduler

# Step decay: multiply lr by gamma every step_size epochs
lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Multi-step: decay at specified milestones
lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

# Exponential: lr *= gamma every epoch
lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# Cosine annealing
lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)

# Cosine with warm restarts
lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0)

# OneCycleLR (1-cycle policy, call every batch)
lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=10)

# Cyclic LR
lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=1e-2,
                       step_size_up=2000, mode='triangular2')

# Reduce on plateau
lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                                threshold=1e-4, min_lr=0, cooldown=0)
# Usage: scheduler.step(val_loss)  # not epoch

# Linear warmup
lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=100)

# Constant
lr_scheduler.ConstantLR(optimizer, factor=0.1, total_iters=5)

# Polynomial decay
lr_scheduler.PolynomialLR(optimizer, total_iters=100, power=1.0)

# Lambda (custom)
lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

# Multiplicative
lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 0.95)

# Chain multiple schedulers sequentially
lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[100])

# Run multiple schedulers simultaneously
lr_scheduler.ChainedScheduler(schedulers=[warmup, cosine])
```

## Linear Warmup + Cosine Decay (common transformer recipe)

```python
warmup = lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=500)
cosine = lr_scheduler.CosineAnnealingLR(optimizer, T_max=9500, eta_min=1e-6)
scheduler = lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[500])

for epoch in range(epochs):
    train_one_epoch(...)
    scheduler.step()
    print(f"lr: {scheduler.get_last_lr()}")
```
