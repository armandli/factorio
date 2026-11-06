# PyTorch Training Patterns Reference

## DataLoader and Dataset

```python
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

# Map-style dataset
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Iterable dataset (for streaming data)
class StreamDataset(IterableDataset):
    def __iter__(self):
        for sample in self.source:
            yield sample

# DataLoader
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,          # parallel workers (0 = main process only)
    pin_memory=True,        # faster CPU→GPU transfer
    drop_last=False,        # drop last incomplete batch
    persistent_workers=True, # keep workers alive between epochs
    prefetch_factor=2,      # batches to prefetch per worker
    collate_fn=None,        # custom batch collation
)

# Quick tensor dataset
dataset = TensorDataset(X, y)

# Train/val split
train_ds, val_ds = random_split(dataset, [0.8, 0.2])
```

## Full Training Loop with Best Practices

Mixed precision (`autocast` + `GradScaler`) is the default here — see
`pytorch-amp-guide-python` for the full AMP API.

```python
def train(model, train_loader, val_loader, epochs, device):
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    scaler = GradScaler(device.type, enabled=(device.type == "cuda"))

    best_val_loss = float("inf")

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        # --- Validate ---
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device.type):
                    logits = model(x)
                    val_loss += loss_fn(logits, y).item()
                correct += (logits.argmax(1) == y).sum().item()

        scheduler.step()
        print(f"Epoch {epoch} | train_loss={total_loss/len(train_loader):.4f} "
              f"val_loss={val_loss/len(val_loader):.4f} "
              f"acc={correct/len(val_loader.dataset):.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")
```

## Mixed Precision (AMP)

Mixed precision is the default training mode in this repo — the loop above
already applies it. For the full `torch.amp` API (op-eligibility tables,
`float16` vs `bfloat16` tradeoffs, gradient accumulation/penalty under
scaling, multiple models/optimizers, DataParallel/DDP, custom autograd
`Function`s, and a real anti-pattern this repo's own notebooks fell into),
see the `pytorch-amp-guide-python` skill.

## torch.compile

```python
# Compile model (traces graph, JIT-compiles with Inductor backend by default)
model = torch.compile(model)

# Modes
model = torch.compile(model, mode="default")          # balanced
model = torch.compile(model, mode="reduce-overhead")  # minimize Python overhead
model = torch.compile(model, mode="max-autotune")     # maximum throughput, slow compile

# Options
model = torch.compile(model, fullgraph=True)   # error on graph breaks (strict)
model = torch.compile(model, dynamic=True)     # support dynamic shapes
model = torch.compile(model, backend="eager")  # disable compilation (debug)

# Compile just a function
fast_fn = torch.compile(my_fn)
```

## Checkpoint Saving and Loading

```python
# Full checkpoint (model + optimizer + epoch)
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "loss": loss,
}, "checkpoint.pt")

ckpt = torch.load("checkpoint.pt", map_location=device, weights_only=True)
model.load_state_dict(ckpt["model_state_dict"])
optimizer.load_state_dict(ckpt["optimizer_state_dict"])
scheduler.load_state_dict(ckpt["scheduler_state_dict"])
start_epoch = ckpt["epoch"] + 1
```

## CUDA Memory Management

```python
device = torch.device("cuda:0")

torch.cuda.is_available()
torch.cuda.device_count()
torch.cuda.current_device()
torch.cuda.set_device(0)

# Memory
torch.cuda.memory_allocated(device)       # bytes used by tensors
torch.cuda.max_memory_allocated(device)   # peak usage
torch.cuda.memory_reserved(device)        # total reserved by caching allocator
torch.cuda.empty_cache()                  # free unoccupied cached memory
torch.cuda.reset_max_memory_allocated()

# Streams
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    y = model(x)
torch.cuda.synchronize()   # wait for all GPU work to finish
```

## Distributed Data Parallel (DDP)

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Initialize process group (called once per process)
dist.init_process_group(backend="nccl")   # nccl for GPU, gloo for CPU
local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device(f"cuda:{local_rank}")

model = MyModel().to(device)
model = DDP(model, device_ids=[local_rank])

# Use DistributedSampler to partition dataset
sampler = DistributedSampler(dataset, shuffle=True)
loader = DataLoader(dataset, batch_size=32, sampler=sampler)

for epoch in range(epochs):
    sampler.set_epoch(epoch)   # reshuffle each epoch
    for x, y in loader:
        ...

dist.destroy_process_group()

# Launch: torchrun --nproc_per_node=4 train.py
```

## Gradient Checkpointing

```python
from torch.utils.checkpoint import checkpoint

class Block(nn.Module):
    def forward(self, x):
        return checkpoint(self._fwd, x, use_reentrant=False)

    def _fwd(self, x):
        return self.layers(x)
```

## Profiling

```python
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    on_trace_ready=tensorboard_trace_handler("./log/profiler"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, (x, y) in enumerate(loader):
        model(x)
        prof.step()
        if step >= 10:
            break

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

## Common Patterns

### Freeze pretrained backbone
```python
for name, param in model.named_parameters():
    if "head" not in name:
        param.requires_grad = False
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
```

### Exponential Moving Average (EMA)
```python
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))
for x, y in loader:
    # ... train model ...
    ema_model.update_parameters(model)
```

### Gradient Accumulation
```python
accumulation_steps = 4
optimizer.zero_grad()
for i, (x, y) in enumerate(loader):
    with autocast(device_type="cuda"):
        loss = loss_fn(model(x), y) / accumulation_steps
    scaler.scale(loss).backward()
    if (i + 1) % accumulation_steps == 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```
