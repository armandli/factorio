# NumPy Array Computing in marimo Notebooks

## Setup

```python
with app.setup:
    import marimo as mo
    import numpy as np
```

Add `numpy` to PEP 723 dependencies:

```python
# /// script
# dependencies = ["marimo", "numpy"]
# ///
```

## Array Creation

```python
np.zeros((3, 4))            # zeros
np.ones((2, 3))             # ones
np.array([1, 2, 3])         # from list
np.arange(0, 10, 2)         # range with step
np.linspace(0, 1, 5)        # evenly spaced
np.eye(3)                   # identity matrix
np.full((2, 3), 7)          # filled with constant
```

## Array Manipulation

```python
arr.reshape((4,))            # change shape
arr.T                        # transpose (swap axes)
np.concatenate([a, b])       # join along existing axis
np.stack([a, b])             # join along new axis
np.split(arr, 2)             # split into parts
np.squeeze(arr)              # remove size-1 axes
np.expand_dims(arr, axis=0)  # add axis
```

## Indexing & Slicing

```python
arr[0]           # first row
arr[:, 0]        # first column
arr[1:3, 2:4]    # subarray
arr[arr > 10]    # boolean indexing
arr[[0, 2], :]   # fancy indexing
```

## Mathematical Operations

```python
np.add(a, b) / a + b        # element-wise addition
np.multiply(a, b) / a * b   # element-wise multiply
np.sqrt(arr)                 # element-wise sqrt
np.exp(arr)                  # e^x
np.log(arr)                  # natural log
np.abs(arr)                  # absolute value
np.clip(arr, 0, 1)           # clamp values
```

## Reductions

```python
np.sum(arr)                  # total sum
np.sum(arr, axis=0)          # column sums
np.mean(arr)                 # mean
np.std(arr)                  # standard deviation
np.min(arr), np.max(arr)     # min/max
np.argmin(arr), np.argmax(arr)  # index of min/max
np.cumsum(arr)               # cumulative sum
```

## Statistics

```python
np.median(arr)
np.percentile(arr, 75)
np.var(arr)
np.cov(arr)                  # covariance matrix
np.corrcoef(arr)             # correlation matrix
```

## Sorting & Searching

```python
np.sort(arr)
np.argsort(arr)              # indices of sorted order
np.unique(arr)               # unique elements
np.where(arr > 5)            # indices where True
np.searchsorted(arr, 5)      # insert position
```

## Linear Algebra (`np.linalg`)

```python
np.dot(a, b)                 # dot product
np.matmul(a, b) / a @ b      # matrix multiply
np.linalg.inv(arr)           # matrix inverse
np.linalg.det(arr)           # determinant
np.linalg.norm(arr)          # norm
U, s, Vt = np.linalg.svd(arr)      # SVD
vals, vecs = np.linalg.eig(arr)    # eigendecomposition
```

## Random Number Generation

```python
# Preferred modern API
rng = np.random.default_rng(seed=42)
rng.random((3, 3))           # uniform [0, 1)
rng.standard_normal((3, 3))  # normal distribution
rng.integers(0, 10, 5)       # random integers
rng.choice([1, 2, 3], 5)     # random sample
rng.shuffle(arr)             # in-place shuffle

# Legacy (still common in existing code)
np.random.seed(42)
np.random.rand(3, 3)
np.random.randn(3, 3)
np.random.randint(0, 10, 5)
```

## Working with Polars and Altair

```python
# Polars column -> NumPy array
arr = np.array(df["col"])
arr = df["col"].to_numpy()

# NumPy array -> Polars Series
series = pl.Series("name", np_array)

# Altair accepts both pandas and Polars DataFrames with NumPy-computed columns
```
