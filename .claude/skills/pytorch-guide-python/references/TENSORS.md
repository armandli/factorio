# PyTorch Tensor Reference

## Data Types

| dtype | Description |
|-------|-------------|
| `torch.float32` / `torch.float` | Default float |
| `torch.float16` / `torch.half` | Half precision |
| `torch.bfloat16` | Brain float (better range than float16) |
| `torch.float64` / `torch.double` | Double precision |
| `torch.int8`, `torch.int16`, `torch.int32`, `torch.int64` | Integer types |
| `torch.uint8`, `torch.uint16`, `torch.uint32`, `torch.uint64` | Unsigned |
| `torch.bool` | Boolean |
| `torch.complex64`, `torch.complex128` | Complex |

## Tensor Creation

```python
# From data
torch.tensor(data, dtype=None, device=None, requires_grad=False)
torch.as_tensor(data, dtype=None, device=None)   # zero-copy when possible
torch.from_numpy(ndarray)                          # shares memory with numpy

# Shape-based
torch.zeros(*size, dtype=None, device=None)
torch.ones(*size, dtype=None, device=None)
torch.full(size, fill_value, dtype=None, device=None)
torch.empty(*size, dtype=None, device=None)        # uninitialized
torch.eye(n, m=None, dtype=None, device=None)

# Range
torch.arange(start, end, step=1, dtype=None, device=None)
torch.linspace(start, end, steps, dtype=None, device=None)
torch.logspace(start, end, steps, base=10.0)

# Like existing tensor (same shape/dtype/device)
torch.zeros_like(input)
torch.ones_like(input)
torch.empty_like(input)
torch.full_like(input, fill_value)
torch.rand_like(input)
torch.randn_like(input)

# Diagonal / triangular
torch.diag(input, diagonal=0)
torch.tril(input, diagonal=0)
torch.triu(input, diagonal=0)
```

## Random Sampling

```python
torch.manual_seed(42)           # reproducibility
torch.cuda.manual_seed(42)      # GPU seed

torch.rand(*size)               # uniform [0, 1)
torch.randn(*size)              # standard normal N(0,1)
torch.randint(low, high, size)  # integer [low, high)
torch.randperm(n)               # random permutation 0..n-1
torch.normal(mean, std, size)   # normal with given mean/std
torch.bernoulli(p)              # Bernoulli from probability tensor
torch.multinomial(weights, num_samples, replacement=False)
```

## Pointwise Math Operations

```python
# Arithmetic
torch.add(a, b, alpha=1)        # a + alpha*b
torch.sub(a, b)
torch.mul(a, b)
torch.div(a, b, rounding_mode=None)   # rounding_mode: None|'floor'|'trunc'
torch.pow(a, exponent)
torch.neg(a)
torch.abs(a)
torch.sign(a)

# Exponential / Log
torch.exp(a);   torch.expm1(a)
torch.log(a);   torch.log2(a);   torch.log10(a);   torch.log1p(a)
torch.sqrt(a);  torch.rsqrt(a);  torch.cbrt(a)

# Trigonometric
torch.sin(a);  torch.cos(a);  torch.tan(a)
torch.asin(a); torch.acos(a); torch.atan(a); torch.atan2(a, b)
torch.sinh(a); torch.cosh(a); torch.tanh(a)
torch.asinh(a);torch.acosh(a);torch.atanh(a)

# Rounding / Clamp
torch.floor(a); torch.ceil(a); torch.round(a, decimals=0)
torch.clamp(a, min=None, max=None)
torch.clip(a, min, max)     # alias for clamp

# Special
torch.sigmoid(a)
torch.erf(a);  torch.erfc(a);  torch.erfinv(a)
torch.nan_to_num(a, nan=0.0, posinf=None, neginf=None)
torch.where(condition, x, y)

# Bitwise
torch.bitwise_and(a, b)
torch.bitwise_or(a, b)
torch.bitwise_xor(a, b)
torch.bitwise_not(a)
torch.bitwise_left_shift(a, n)
torch.bitwise_right_shift(a, n)
```

## Reduction Operations

```python
torch.sum(a, dim=None, keepdim=False)
torch.mean(a, dim=None, keepdim=False)
torch.var(a, dim=None, correction=1, keepdim=False)
torch.std(a, dim=None, correction=1, keepdim=False)
torch.prod(a, dim=None, keepdim=False)
torch.min(a, dim=None, keepdim=False)    # returns (values, indices) if dim set
torch.max(a, dim=None, keepdim=False)
torch.argmin(a, dim=None, keepdim=False)
torch.argmax(a, dim=None, keepdim=False)
torch.median(a, dim=None, keepdim=False)
torch.logsumexp(a, dim, keepdim=False)
torch.any(a, dim=None, keepdim=False)
torch.all(a, dim=None, keepdim=False)
torch.cumsum(a, dim)
torch.cumprod(a, dim)
```

## Shape Manipulation

```python
a.reshape(*shape)          # may copy
a.view(*shape)             # zero-copy (must be contiguous)
a.contiguous()             # make contiguous
a.squeeze(dim=None)        # remove size-1 dims
a.unsqueeze(dim)           # add size-1 dim
a.permute(*dims)           # reorder all dims
a.transpose(dim0, dim1)    # swap two dims
a.T                        # reverse all dims (2D: transpose)
a.mT                       # transpose last two dims
a.flatten(start_dim=0, end_dim=-1)
a.unflatten(dim, sizes)
a.expand(*sizes)           # broadcast (no memory copy)
a.repeat(*repeats)         # actual copy
a.tile(dims)               # alias torch.tile

torch.reshape(a, shape)
torch.squeeze(a, dim=None)
torch.unsqueeze(a, dim)
torch.permute(a, dims)
torch.transpose(a, dim0, dim1)
torch.flatten(a, start_dim=0, end_dim=-1)
```

## Joining / Splitting

```python
torch.cat(tensors, dim=0)              # concatenate along dim
torch.stack(tensors, dim=0)            # new dim
torch.hstack(tensors)                  # horizontal (dim=1)
torch.vstack(tensors)                  # vertical (dim=0)
torch.dstack(tensors)                  # depth (dim=2)

torch.split(tensor, split_size_or_sections, dim=0)
torch.chunk(tensor, chunks, dim=0)
torch.unbind(tensor, dim=0)            # tuple of slices

torch.broadcast_tensors(*tensors)
torch.broadcast_to(tensor, shape)
```

## Indexing / Selection

```python
a[0, :, 2]                      # standard indexing
a[[0, 2], :]                     # fancy indexing

torch.gather(a, dim, index)      # gather along dim using index
torch.scatter(a, dim, index, src)
torch.scatter_add(a, dim, index, src)

torch.index_select(a, dim, index)
torch.masked_select(a, mask)     # 1-D result
torch.take(a, index)             # flat indexing
torch.take_along_dim(a, indices, dim)

torch.nonzero(a, as_tuple=False)
torch.topk(a, k, dim=-1, largest=True, sorted=True)
torch.kthvalue(a, k, dim=-1)
torch.sort(a, dim=-1, descending=False)
torch.argsort(a, dim=-1, descending=False)

torch.unique(a, sorted=True, return_inverse=False, return_counts=False, dim=None)
```

## Matrix / Linear Algebra

```python
# Matrix multiply
torch.mm(a, b)           # 2D matrix multiply
torch.bmm(a, b)          # batched matrix multiply (3D)
torch.matmul(a, b)       # @ operator, broadcasts
torch.einsum(equation, *operands)
torch.inner(a, b)
torch.outer(a, b)
torch.tensordot(a, b, dims)
torch.addmm(c, a, b, alpha=1, beta=1)  # beta*c + alpha*(a@b)

# torch.linalg
torch.linalg.norm(a, ord=None, dim=None, keepdim=False)
torch.linalg.vector_norm(a, ord=2, dim=None, keepdim=False)
torch.linalg.matrix_norm(a, ord='fro', dim=(-2,-1), keepdim=False)
torch.linalg.det(a)
torch.linalg.slogdet(a)
torch.linalg.matrix_rank(a, atol=None, rtol=None)
torch.linalg.cond(a, p=None)
torch.linalg.inv(a)
torch.linalg.pinv(a, atol=None, rtol=None)
torch.linalg.solve(a, b)
torch.linalg.solve_triangular(a, b, upper=True)
torch.linalg.lstsq(a, b, rcond=None)
torch.linalg.cholesky(a, upper=False)
torch.linalg.qr(a, mode='reduced')
torch.linalg.svd(a, full_matrices=True)
torch.linalg.svdvals(a)
torch.linalg.eig(a)
torch.linalg.eigvals(a)
torch.linalg.eigh(a, UPLO='L')
torch.linalg.eigvalsh(a, UPLO='L')
torch.linalg.lu(a, pivot=True)
torch.linalg.lu_factor(a, pivot=True)
torch.linalg.cross(a, b, dim=-1)
torch.linalg.multi_dot(tensors)
torch.linalg.matrix_exp(a)
torch.linalg.matrix_power(a, n)
torch.linalg.vander(x, N=None)
```

## FFT

```python
import torch.fft

torch.fft.fft(input, n=None, dim=-1, norm=None)
torch.fft.ifft(input, n=None, dim=-1, norm=None)
torch.fft.fft2(input, s=None, dim=(-2,-1), norm=None)
torch.fft.ifft2(input, s=None, dim=(-2,-1), norm=None)
torch.fft.fftn(input, s=None, dim=None, norm=None)
torch.fft.ifftn(input, s=None, dim=None, norm=None)
torch.fft.rfft(input, n=None, dim=-1, norm=None)
torch.fft.irfft(input, n=None, dim=-1, norm=None)
torch.fft.fftfreq(n, d=1.0, device=None)
torch.fft.rfftfreq(n, d=1.0, device=None)
torch.fft.fftshift(input, dim=None)
torch.fft.ifftshift(input, dim=None)
```

## Comparison

```python
torch.eq(a, b);  torch.ne(a, b)
torch.gt(a, b);  torch.ge(a, b)
torch.lt(a, b);  torch.le(a, b)
torch.equal(a, b)       # True if same shape and all elements equal
torch.allclose(a, b, atol=1e-8, rtol=1e-5)
torch.isnan(a);  torch.isinf(a);  torch.isfinite(a)
torch.isclose(a, b, atol=1e-8, rtol=1e-5)
torch.maximum(a, b);  torch.minimum(a, b)
```

## Serialization

```python
torch.save(obj, "path.pt")            # pickle-based save
obj = torch.load("path.pt", map_location=device, weights_only=True)

# Recommended for state dicts (weights_only=True for safety)
torch.save(model.state_dict(), "weights.pt")
model.load_state_dict(torch.load("weights.pt", weights_only=True))
```
