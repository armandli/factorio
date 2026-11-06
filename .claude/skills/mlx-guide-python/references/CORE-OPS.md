# MLX Core Operations Reference

All ops are in `mlx.core` (imported as `mx`). All support an optional `stream=` parameter.

## Array Creation

| Function | Description |
|----------|-------------|
| `mx.array(data, dtype=None)` | Create from Python list/numpy array |
| `mx.zeros(shape, dtype=mx.float32)` | All zeros |
| `mx.ones(shape, dtype=mx.float32)` | All ones |
| `mx.full(shape, fill_value, dtype=None)` | Fill with value |
| `mx.eye(n, m=None, dtype=mx.float32)` | Identity matrix |
| `mx.identity(n, dtype=mx.float32)` | Square identity |
| `mx.arange(start, stop, step, dtype=None)` | Evenly spaced values |
| `mx.linspace(start, stop, num, dtype=mx.float32)` | Evenly spaced over interval |
| `mx.zeros_like(a)` | Zeros with same shape/dtype |
| `mx.ones_like(a)` | Ones with same shape/dtype |
| `mx.diag(a, k=0)` | Extract diagonal or construct diagonal matrix |
| `mx.tri(n, m=None, k=0, dtype=mx.float32)` | Lower triangle of ones |
| `mx.tril(a, k=0)` | Lower triangle |
| `mx.triu(a, k=0)` | Upper triangle |

## Data Types

`mx.bool_`, `mx.uint8`, `mx.uint16`, `mx.uint32`, `mx.uint64`,
`mx.int8`, `mx.int16`, `mx.int32`, `mx.int64`,
`mx.float16`, `mx.bfloat16`, `mx.float32`, `mx.complex64`

```python
x.dtype          # inspect dtype
x.astype(mx.float16)
mx.issubdtype(x.dtype, mx.floating)
mx.finfo(mx.float32)   # dtype info (eps, max, etc.)
```

## Arithmetic

| Op | Operator | Description |
|----|----------|-------------|
| `mx.add(a, b)` | `a + b` | Element-wise add |
| `mx.subtract(a, b)` | `a - b` | Element-wise subtract |
| `mx.multiply(a, b)` | `a * b` | Element-wise multiply |
| `mx.divide(a, b)` | `a / b` | Element-wise divide |
| `mx.power(a, b)` | `a ** b` | Element-wise power |
| `mx.negative(a)` | `-a` | Negate |
| `mx.floor_divide(a, b)` | `a // b` | Floor divide |
| `mx.remainder(a, b)` | `a % b` | Remainder |
| `mx.abs(a)` | `abs(a)` | Absolute value |
| `mx.sign(a)` | — | Sign (-1, 0, 1) |

## Comparison & Logical

```python
mx.equal(a, b)        # a == b
mx.not_equal(a, b)    # a != b
mx.greater(a, b)      # a > b
mx.greater_equal(a, b)
mx.less(a, b)
mx.less_equal(a, b)
mx.logical_and(a, b)
mx.logical_or(a, b)
mx.logical_not(a)
mx.allclose(a, b, atol=1e-8, rtol=1e-5)
mx.isclose(a, b)
mx.isnan(a)
mx.isinf(a)
mx.isfinite(a)
```

## Bitwise

```python
mx.bitwise_and(a, b)   # a & b
mx.bitwise_or(a, b)    # a | b
mx.bitwise_xor(a, b)   # a ^ b
mx.bitwise_invert(a)   # ~a
mx.left_shift(a, n)    # a << n
mx.right_shift(a, n)   # a >> n
```

## Math / Transcendental

```python
# Exponential / Log
mx.exp(a)
mx.expm1(a)      # exp(a) - 1, more accurate near 0
mx.log(a)
mx.log2(a)
mx.log10(a)
mx.log1p(a)      # log(1 + a), more accurate near 0
mx.sigmoid(a)    # 1 / (1 + exp(-a))
mx.erf(a)        # error function

# Trigonometric
mx.sin(a);  mx.cos(a);  mx.tan(a)
mx.arcsin(a);  mx.arccos(a);  mx.arctan(a)
mx.arctan2(a, b)

# Hyperbolic
mx.sinh(a);  mx.cosh(a);  mx.tanh(a)
mx.arcsinh(a);  mx.arccosh(a);  mx.arctanh(a)

# Rounding
mx.floor(a);  mx.ceil(a);  mx.round(a, decimals=0)

# Sqrt / Cbrt / Rsqrt
mx.sqrt(a);  mx.rsqrt(a);  mx.cbrt(a)
```

## Reductions

```python
mx.sum(a, axis=None, keepdims=False)
mx.prod(a, axis=None, keepdims=False)
mx.mean(a, axis=None, keepdims=False)
mx.var(a, axis=None, ddof=0, keepdims=False)
mx.std(a, axis=None, ddof=0, keepdims=False)
mx.min(a, axis=None, keepdims=False)
mx.max(a, axis=None, keepdims=False)
mx.median(a, axis=None, keepdims=False)
mx.logsumexp(a, axis=None, keepdims=False)
mx.all(a, axis=None, keepdims=False)
mx.any(a, axis=None, keepdims=False)
```

## Cumulative

```python
mx.cumsum(a, axis=None, inclusive=True, reverse=False)
mx.cumprod(a, axis=None, inclusive=True, reverse=False)
mx.cummin(a, axis=None, inclusive=True, reverse=False)
mx.cummax(a, axis=None, inclusive=True, reverse=False)
mx.logcumsumexp(a, axis=None, inclusive=True, reverse=False)
```

## Shape Manipulation

```python
mx.reshape(a, shape)
mx.flatten(a, start_axis=0, end_axis=-1)
mx.unflatten(a, axis, shape)
mx.squeeze(a, axis=None)
mx.expand_dims(a, axis)
mx.transpose(a, axes=None)
mx.swapaxes(a, axis1, axis2)
mx.moveaxis(a, source, destination)
mx.atleast_1d(a);  mx.atleast_2d(a);  mx.atleast_3d(a)
```

## Joining / Splitting

```python
mx.concatenate([a, b, ...], axis=0)
mx.stack([a, b, ...], axis=0)
mx.tile(a, reps)
mx.repeat(a, repeats, axis=None)
mx.pad(a, pad_width, mode="constant", constant_values=0)
mx.split(a, indices_or_sections, axis=0)
```

## Indexing / Selection

```python
mx.take(a, indices, axis=None)
mx.take_along_axis(a, indices, axis)
mx.where(condition, x, y)
mx.clip(a, a_min=None, a_max=None)
mx.slice(a, start, stop, strides)
mx.slice_update(a, update, start, stop, strides)
```

## Sorting

```python
mx.sort(a, axis=-1)
mx.argsort(a, axis=-1)
mx.argmax(a, axis=None, keepdims=False)
mx.argmin(a, axis=None, keepdims=False)
mx.topk(a, k, axis=-1)
mx.partition(a, kth, axis=-1)
mx.argpartition(a, kth, axis=-1)
```

## Matrix / Linear Algebra

```python
# Core
mx.matmul(a, b)          # @ operator
mx.inner(a, b)
mx.outer(a, b)
mx.tensordot(a, b, axes)
mx.einsum(subscripts, *operands)
mx.addmm(c, a, b, alpha=1.0, beta=1.0)  # c = alpha*a@b + beta*c

# linalg submodule
mx.linalg.norm(a, ord=None, axis=None, keepdims=False)
mx.linalg.qr(a, mode="reduced")
mx.linalg.svd(a, compute_uv=True)
mx.linalg.eig(a)
mx.linalg.eigh(a)
mx.linalg.inv(a)
mx.linalg.cholesky(a, upper=False)
mx.linalg.solve(a, b)
mx.linalg.solve_triangular(a, b, upper=False)
mx.linalg.lu(a)
mx.linalg.cross(a, b, axis=-1)
mx.linalg.trace(a, offset=0, axis1=0, axis2=1)
```

## Convolutions

```python
mx.conv1d(input, weight, stride=1, padding=0, dilation=1, groups=1)
mx.conv2d(input, weight, stride=1, padding=0, dilation=1, groups=1)
mx.conv3d(input, weight, stride=1, padding=0, dilation=1, groups=1)
mx.conv_transpose1d(input, weight, stride=1, padding=0, dilation=1)
mx.conv_transpose2d(input, weight, stride=1, padding=0, dilation=1)
mx.conv_transpose3d(input, weight, stride=1, padding=0, dilation=1)
mx.conv_general(input, weight, stride, padding, kernel_dilation, input_dilation, groups, flip)
```

Input layout: `(N, H, W, C)` — **channels last** (different from PyTorch).

## FFT

```python
import mlx.core.fft as fft

fft.fft(a, n=None, axis=-1)
fft.ifft(a, n=None, axis=-1)
fft.fft2(a, s=None, axes=(-2, -1))
fft.ifft2(a, s=None, axes=(-2, -1))
fft.fftn(a, s=None, axes=None)
fft.ifftn(a, s=None, axes=None)
fft.rfft(a, n=None, axis=-1)
fft.irfft(a, n=None, axis=-1)
fft.fftfreq(n, d=1.0)
fft.fftshift(a, axes=None)
fft.ifftshift(a, axes=None)
```

## Quantization

```python
mx.quantize(w, group_size=64, bits=4)           # returns (wq, scales, biases)
mx.dequantize(wq, scales, biases, group_size, bits)
mx.quantized_matmul(x, wq, scales, biases, transpose=True, group_size=64, bits=4)
mx.gather_qmm(x, wq, scales, biases, lhs_indices=None, rhs_indices=None, ...)
```

## I/O

```python
mx.save("file.npy", array)
mx.savez("file.npz", a=arr1, b=arr2)
mx.savez_compressed("file.npz", a=arr1)
mx.save_safetensors("file.safetensors", {"weight": arr})
mx.save_gguf("file.gguf", {"weight": arr}, metadata={})
data = mx.load("file.npz")   # returns dict
```

## Utilities

```python
mx.nan_to_num(a, nan=0.0, posinf=None, neginf=None)
mx.broadcast_to(a, shape)
mx.broadcast_arrays(a, b)
mx.meshgrid(*arrays, indexing="xy")
mx.kron(a, b)
mx.view(a, dtype)            # reinterpret bits
mx.as_strided(a, shape, strides, offset=0)
mx.roll(a, shift, axis=None)
mx.block_masked_mm(a, b, mask_out=None, mask_lhs=None, mask_rhs=None, block_size=32)
mx.hadamard_transform(a, scale=None)
```
