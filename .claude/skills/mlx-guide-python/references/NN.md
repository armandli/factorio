# MLX Neural Network Reference

All in `mlx.nn` (imported as `nn`).

## Module Base Class

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Assign nn.Module or mx.array attributes — MLX tracks them automatically

    def __call__(self, x):
        # forward pass
        return x

model = MyModel()
mx.eval(model.parameters())          # initialize (allocates memory)
params = model.parameters()          # nested dict of arrays
model.update(new_params)             # replace parameters
model.apply(lambda x: x * 2)        # map fn over all params in-place
model.train()                        # set training mode (affects Dropout, BatchNorm)
model.eval()                         # set eval mode
model.freeze()                       # exclude all from grad
model.freeze(keys=["embed"])         # freeze specific submodules
model.unfreeze()
trainable = model.trainable_parameters()
```

## Linear Layers

```python
nn.Linear(input_dims, output_dims, bias=True)
nn.Bilinear(input_dims_1, input_dims_2, output_dims, bias=True)
nn.Embedding(num_embeddings, dims)
# Quantized (4-bit or 8-bit weights, for inference efficiency)
nn.QuantizedLinear(input_dims, output_dims, bias=True, group_size=64, bits=4)
nn.QuantizedEmbedding(num_embeddings, dims, group_size=64, bits=4)
```

## Convolutional Layers

MLX uses **channels-last** layout: `(N, H, W, C)` — opposite of PyTorch.

```python
nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True)
nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True)
nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True)
```

## Normalization

```python
nn.LayerNorm(dims, eps=1e-5, affine=True, bias=True)
nn.RMSNorm(dims, eps=1e-5)
nn.GroupNorm(num_groups, dims, eps=1e-5, affine=True, pytorch_compatible=False)
nn.BatchNorm(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
nn.InstanceNorm(dims, eps=1e-5, affine=False)
```

## Recurrent Layers

```python
nn.RNN(input_size, hidden_size, bias=True, nonlinearity="tanh")
nn.GRU(input_size, hidden_size, bias=True)
nn.LSTM(input_size, hidden_size, bias=True)
# Returns: output, (h_n, c_n) for LSTM; output, h_n for GRU/RNN
```

## Attention

```python
nn.MultiHeadAttention(
    dims,               # model dimension
    num_heads,
    query_input_dims=None,
    key_input_dims=None,
    value_input_dims=None,
    value_dims=None,
    value_output_dims=None,
    bias=False,
)
# call: attn(queries, keys, values, mask=None)

nn.Transformer(
    dims=512,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    mlp_dims=None,
    dropout=0.0,
    activation=nn.relu,
    norm_first=True,
)
```

## Positional Encodings

```python
nn.RoPE(dims, traditional=False, base=10000, scale=1.0)
# usage: x = rope(x, offset=0)

nn.SinusoidalPositionalEncoding(dims, min_freq=1e-4, scale=1.0, cos_first=False, full_turns=False)

nn.ALiBi()    # Attention with Linear Biases
```

## Pooling

```python
nn.MaxPool1d(kernel_size, stride=None, padding=0)
nn.MaxPool2d(kernel_size, stride=None, padding=0)
nn.MaxPool3d(kernel_size, stride=None, padding=0)
nn.AvgPool1d(kernel_size, stride=None, padding=0)
nn.AvgPool2d(kernel_size, stride=None, padding=0)
nn.AvgPool3d(kernel_size, stride=None, padding=0)
```

## Dropout

```python
nn.Dropout(p=0.5)      # 1D
nn.Dropout2d(p=0.5)    # 2D spatial (channels-last)
nn.Dropout3d(p=0.5)    # 3D spatial
```

Active only when `model.train()` is set.

## Activation Functions

```python
# Functional (preferred for one-off use)
nn.relu(x)
nn.gelu(x)               # GeLU (approximate by default)
nn.gelu_approx(x)        # tanh approximation
nn.gelu_fast_approx(x)   # sigmoid approximation, fastest
nn.silu(x)               # Swish = x * sigmoid(x)
nn.sigmoid(x)
nn.tanh(x)
nn.softmax(x, axis=-1)
nn.log_softmax(x, axis=-1)
nn.softplus(x)
nn.softsign(x)
nn.leaky_relu(x, negative_slope=0.01)
nn.elu(x, alpha=1.0)
nn.selu(x)
nn.mish(x)
nn.hardswish(x)
nn.celu(x, alpha=1.0)
nn.prelu(x, weight)
nn.glu(x, axis=-1)       # Gated Linear Unit
nn.step(x, threshold=0.0)

# As Module (for use inside nn.Sequential or assigned to self)
nn.ReLU()
nn.GELU(approx="none")   # "none" | "fast" | "precise"
nn.SiLU()
nn.Tanh()
nn.Sigmoid()
nn.Softmax(axis=-1)
nn.LeakyReLU(negative_slope=0.01)
nn.ELU(alpha=1.0)
nn.SELU()
nn.Mish()
nn.Hardswish()
nn.CELU(alpha=1.0)
nn.PReLU(num_parameters=1, init=0.25)
nn.GLU(axis=-1)
```

## Container

```python
nn.Sequential(*layers)
# forward: applies layers in order
model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
```

## Distributed (Multi-GPU)

```python
import mlx.core.distributed as dist

world = dist.init()          # initialize process group
nn.DistributedLinear(...)    # data-parallel linear
nn.SyncBatchNorm(...)        # synchronized batch norm
dist.all_reduce(x)           # sum across all processes
dist.all_gather(x)
```

## Loss Functions

All in `mlx.nn.losses`. Most accept `reduction="none"|"mean"|"sum"`.

```python
nn.losses.cross_entropy(logits, targets, weights=None, axis=-1, label_smoothing=0.0, reduction="none")
nn.losses.binary_cross_entropy(inputs, targets, weights=None, with_logits=True, reduction="mean")
nn.losses.mse_loss(inputs, targets, reduction="mean")
nn.losses.l1_loss(inputs, targets, reduction="mean")
nn.losses.smooth_l1_loss(inputs, targets, beta=1.0, reduction="mean")
nn.losses.huber_loss(inputs, targets, delta=1.0, reduction="none")
nn.losses.nll_loss(inputs, targets, axis=-1, reduction="mean")
nn.losses.gaussian_nll_loss(inputs, targets, vars, full=False, eps=1e-6, reduction="mean")
nn.losses.kl_div_loss(inputs, targets, axis=-1, reduction="mean")
nn.losses.hinge_loss(inputs, targets, reduction="mean")
nn.losses.cosine_similarity_loss(x1, x2, axis=-1, eps=1e-8, reduction="mean")
nn.losses.triplet_loss(anchors, positives, negatives, margin=1.0, p=2.0, eps=1e-6, reduction="none")
nn.losses.log_cosh_loss(inputs, targets, reduction="mean")
nn.losses.margin_ranking_loss(inputs1, inputs2, targets, margin=0.0, reduction="mean")
```

## Gradient Helper

```python
# Preferred over mx.grad for models (handles non-array params in Module)
loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
loss, grads = loss_and_grad_fn(model, *args)
```
