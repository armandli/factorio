# PyTorch torch.nn Reference

All in `torch.nn` (imported as `nn`).

## Module Base Class

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)      # registered automatically
        self.register_buffer("running_mean", torch.zeros(5))   # non-param tensor

    def forward(self, x):
        return self.fc(x)

model = MyModel()

# Key methods
model.parameters()                  # iterator over all nn.Parameter
model.named_parameters()            # (name, param) iterator
model.state_dict()                  # OrderedDict of all param/buffer tensors
model.load_state_dict(state, strict=True)
model.to(device)                    # move all params/buffers to device
model.train()                       # set training mode (affects Dropout, BN)
model.eval()                        # set eval mode
model.zero_grad(set_to_none=True)   # clear all gradients

# Freeze / unfreeze
for p in model.parameters():
    p.requires_grad = False

# Apply fn recursively to all modules
model.apply(fn)

# Children / modules
list(model.children())             # direct children only
list(model.modules())              # all submodules recursively
```

## Linear Layers

```python
nn.Linear(in_features, out_features, bias=True)
nn.Bilinear(in1_features, in2_features, out_features, bias=True)
nn.LazyLinear(out_features, bias=True)   # infers in_features on first call
```

## Embedding Layers

```python
nn.Embedding(num_embeddings, embedding_dim, padding_idx=None,
             max_norm=None, norm_type=2.0, scale_grad_by_freq=False,
             sparse=False)
nn.EmbeddingBag(num_embeddings, embedding_dim, max_norm=None,
                norm_type=2.0, scale_grad_by_freq=False,
                mode='mean', sparse=False, padding_idx=None)
```

## Convolutional Layers

PyTorch uses **channels-first** layout: `(N, C, H, W)`.

```python
nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0,
          dilation=1, groups=1, bias=True, padding_mode='zeros')
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
          dilation=1, groups=1, bias=True, padding_mode='zeros')
nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0,
          dilation=1, groups=1, bias=True)
nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=1,
                   padding=0, output_padding=0, groups=1, bias=True, dilation=1)
nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1,
                   padding=0, output_padding=0, groups=1, bias=True, dilation=1)
nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=1,
                   padding=0, output_padding=0, groups=1, bias=True, dilation=1)

# Lazy variants (auto-infer in_channels)
nn.LazyConv1d / LazyConv2d / LazyConv3d
nn.LazyConvTranspose1d / LazyConvTranspose2d / LazyConvTranspose3d
```

## Pooling

```python
nn.MaxPool1d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
nn.MaxPool3d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
nn.AvgPool1d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
nn.AvgPool3d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
nn.AdaptiveMaxPool1d(output_size, return_indices=False)
nn.AdaptiveMaxPool2d(output_size, return_indices=False)
nn.AdaptiveAvgPool1d(output_size)
nn.AdaptiveAvgPool2d(output_size)
nn.AdaptiveAvgPool3d(output_size)
nn.MaxUnpool1d / MaxUnpool2d / MaxUnpool3d
nn.FractionalMaxPool2d / FractionalMaxPool3d
nn.LPPool1d / LPPool2d / LPPool3d
```

## Normalization

```python
nn.BatchNorm1d(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
nn.BatchNorm2d(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
nn.BatchNorm3d(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
nn.LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, bias=True)
nn.RMSNorm(normalized_shape, eps=1e-8, elementwise_affine=True)
nn.GroupNorm(num_groups, num_channels, eps=1e-5, affine=True)
nn.InstanceNorm1d(num_features, eps=1e-5, momentum=0.1, affine=False)
nn.InstanceNorm2d(num_features, eps=1e-5, momentum=0.1, affine=False)
nn.InstanceNorm3d(num_features, eps=1e-5, momentum=0.1, affine=False)
nn.LocalResponseNorm(size, alpha=1e-4, beta=0.75, k=1.0)
```

## Recurrent Layers

```python
nn.RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh',
       bias=True, batch_first=False, dropout=0.0, bidirectional=False)
nn.LSTM(input_size, hidden_size, num_layers=1,
        bias=True, batch_first=False, dropout=0.0, bidirectional=False,
        proj_size=0)
nn.GRU(input_size, hidden_size, num_layers=1,
       bias=True, batch_first=False, dropout=0.0, bidirectional=False)

# Cell variants (single step)
nn.RNNCell(input_size, hidden_size, bias=True, nonlinearity='tanh')
nn.LSTMCell(input_size, hidden_size, bias=True)
nn.GRUCell(input_size, hidden_size, bias=True)

# Usage (LSTM)
lstm = nn.LSTM(10, 20, 2, batch_first=True)
out, (h_n, c_n) = lstm(x)   # x: (batch, seq, input_size) with batch_first=True
```

## Attention and Transformer

```python
nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=True,
                      add_bias_kv=False, add_zero_attn=False,
                      kdim=None, vdim=None, batch_first=False)
# output: attn_output, attn_weights = attn(query, key, value, key_padding_mask, need_weights, attn_mask)

nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
               dim_feedforward=2048, dropout=0.1, activation='relu',
               batch_first=False, norm_first=False)
nn.TransformerEncoder(encoder_layer, num_layers, norm=None)
nn.TransformerDecoder(decoder_layer, num_layers, norm=None)
nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1,
                            activation='relu', batch_first=False, norm_first=False)
nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1,
                            activation='relu', batch_first=False, norm_first=False)
```

## Dropout

```python
nn.Dropout(p=0.5, inplace=False)
nn.Dropout1d(p=0.5, inplace=False)   # zero out entire channels
nn.Dropout2d(p=0.5, inplace=False)
nn.Dropout3d(p=0.5, inplace=False)
nn.AlphaDropout(p=0.5, inplace=False)  # for SELU networks
nn.FeatureAlphaDropout(p=0.5, inplace=False)
```

## Activation Functions (nn module form)

```python
nn.ReLU(inplace=False)
nn.ReLU6(inplace=False)
nn.LeakyReLU(negative_slope=0.01, inplace=False)
nn.PReLU(num_parameters=1, init=0.25)
nn.ELU(alpha=1.0, inplace=False)
nn.SELU(inplace=False)
nn.CELU(alpha=1.0, inplace=False)
nn.GELU(approximate='none')      # 'none' | 'tanh'
nn.SiLU(inplace=False)           # Swish = x * sigmoid(x)
nn.Mish(inplace=False)
nn.Hardswish(inplace=False)
nn.Hardsigmoid(inplace=False)
nn.Sigmoid()
nn.Tanh()
nn.Softmax(dim=None)
nn.LogSoftmax(dim=None)
nn.Softplus(beta=1, threshold=20)
nn.Softsign()
nn.Hardtanh(min_val=-1.0, max_val=1.0, inplace=False)
nn.Threshold(threshold, value, inplace=False)
nn.GLU(dim=-1)
nn.Tanhshrink()

# Functional equivalents (no state, use in forward directly)
F.relu(x, inplace=False)
F.gelu(x, approximate='none')
F.silu(x, inplace=False)
F.sigmoid(x)
F.tanh(x)
F.softmax(x, dim=-1)
F.log_softmax(x, dim=-1)
F.leaky_relu(x, negative_slope=0.01, inplace=False)
F.elu(x, alpha=1.0, inplace=False)
```

## Containers

```python
nn.Sequential(*modules)                  # forward applies modules in order
nn.ModuleList(modules=None)              # list of modules (registered)
nn.ModuleDict(modules=None)              # dict of modules (registered)
nn.ParameterList(parameters=None)
nn.ParameterDict(parameters=None)

# Example Sequential
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 10),
)
```

## Loss Functions

All accept `reduction='none'|'mean'|'sum'` unless noted.

```python
nn.CrossEntropyLoss(weight=None, ignore_index=-100, reduction='mean',
                    label_smoothing=0.0)
nn.NLLLoss(weight=None, ignore_index=-100, reduction='mean')
nn.BCELoss(weight=None, reduction='mean')
nn.BCEWithLogitsLoss(weight=None, reduction='mean', pos_weight=None)
nn.MSELoss(reduction='mean')
nn.L1Loss(reduction='mean')
nn.SmoothL1Loss(reduction='mean', beta=1.0)
nn.HuberLoss(reduction='mean', delta=1.0)
nn.KLDivLoss(reduction='mean', log_target=False)
nn.HingeEmbeddingLoss(margin=1.0, reduction='mean')
nn.MarginRankingLoss(margin=0.0, reduction='mean')
nn.TripletMarginLoss(margin=1.0, p=2.0, eps=1e-6, swap=False, reduction='mean')
nn.TripletMarginWithDistanceLoss(distance_function=None, margin=1.0, swap=False, reduction='mean')
nn.CosineEmbeddingLoss(margin=0.0, reduction='mean')
nn.MultiLabelMarginLoss(reduction='mean')
nn.MultiLabelSoftMarginLoss(weight=None, reduction='mean')
nn.MultiMarginLoss(p=1, margin=1.0, weight=None, reduction='mean')
nn.GaussianNLLLoss(full=False, eps=1e-6, reduction='mean')
nn.PoissonNLLLoss(log_input=True, full=False, eps=1e-8, reduction='mean')
nn.CTCLoss(blank=0, reduction='mean', zero_infinity=False)
```

## Utilities

```python
# Gradient clipping
nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=2.0)
nn.utils.clip_grad_value_(parameters, clip_value)

# Weight initialization
nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
nn.init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
nn.init.xavier_normal_(tensor, gain=1.0)
nn.init.xavier_uniform_(tensor, gain=1.0)
nn.init.normal_(tensor, mean=0.0, std=1.0)
nn.init.constant_(tensor, val)
nn.init.zeros_(tensor)
nn.init.ones_(tensor)
nn.init.orthogonal_(tensor, gain=1)
nn.init.trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0)

# Spectral norm, weight norm
nn.utils.spectral_norm(module, name='weight', n_power_iterations=1)
nn.utils.weight_norm(module, name='weight', dim=0)
nn.utils.remove_weight_norm(module, name='weight')

# Parameter flattening (for custom optimizers)
nn.utils.parameters_to_vector(parameters)
nn.utils.vector_to_parameters(vec, parameters)
```
