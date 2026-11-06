# PyTorch Neural Networks in marimo Notebooks

## Setup

```python
with app.setup:
    import marimo as mo
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
```

Add `torch` to PEP 723 dependencies:

```python
# /// script
# dependencies = ["marimo", "torch"]
# ///
```

## Device Selection

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Apple Silicon (MPS):
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

## Tensor Creation

```python
torch.tensor([1.0, 2.0, 3.0])              # from list
torch.zeros(3, 4)                            # zeros
torch.ones(2, 3)                             # ones
torch.randn(3, 3)                            # normal distribution
torch.arange(0, 10, 2, dtype=torch.float32)
torch.from_numpy(np_array)                   # from numpy

# Move to device
x = x.to(device)
```

## Tensor Operations

```python
# Arithmetic
a + b, a - b, a * b, a / b
torch.matmul(a, b) / a @ b    # matrix multiply
torch.dot(a, b)                # dot product

# Shape
a.shape, a.dtype, a.device
a.reshape(2, 3)
a.squeeze(), a.unsqueeze(0)
a.permute(1, 0, 2)             # reorder axes
torch.cat([a, b], dim=0)       # concatenate along existing dim
torch.stack([a, b], dim=0)     # new dimension

# Reduction
a.sum(), a.mean(), a.std()
a.sum(dim=0)                   # along axis
a.max(), a.argmax()
```

## Autograd

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x
y.backward()
print(x.grad)   # dy/dx = 2x + 3 at x=2 -> 7.0

# Disable grad for inference
with torch.no_grad():
    predictions = model(test_data)
```

## Defining Models (`nn.Module`)

```python
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)

model = MLP(10, 64, 1).to(device)
```

## Common Layers

```python
nn.Linear(in_features, out_features)       # fully connected
nn.Conv2d(in_channels, out_channels, kernel_size)  # 2D convolution
nn.BatchNorm1d(num_features)               # batch normalization
nn.LayerNorm(normalized_shape)             # layer normalization
nn.Dropout(p=0.2)                          # dropout
nn.Embedding(vocab_size, embedding_dim)    # embedding
nn.LSTM(input_size, hidden_size, num_layers)  # recurrent
nn.TransformerEncoder(encoder_layer, num_layers)  # transformer encoder

# Activations
nn.ReLU(), nn.Sigmoid(), nn.Tanh(), nn.GELU()
```

## Loss Functions

```python
nn.MSELoss()              # regression (mean squared error)
nn.CrossEntropyLoss()     # multi-class classification (includes softmax)
nn.BCEWithLogitsLoss()    # binary classification (includes sigmoid; preferred over BCELoss)
nn.L1Loss()               # mean absolute error
nn.NLLLoss()              # negative log-likelihood
```

## Optimizers

```python
optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
scheduler.step()  # call after each epoch
```

## Training Loop Pattern

```python
model.train()
for epoch in range(num_epochs):
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
    scheduler.step()

# Evaluation
model.eval()
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        pred = model(X_batch.to(device))
        # compute metrics...
```

## Dataset and DataLoader

```python
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

loader = DataLoader(MyDataset(X, y), batch_size=32, shuffle=True)
```

## Save & Load

```python
# Save
torch.save(model.state_dict(), "model.pth")

# Load
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()
```

## NumPy Interop

```python
# Tensor -> NumPy (must be on CPU)
arr = tensor.detach().cpu().numpy()

# NumPy -> Tensor
tensor = torch.from_numpy(arr).float()
```
