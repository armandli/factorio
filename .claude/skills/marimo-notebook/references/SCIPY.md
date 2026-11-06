# SciPy Scientific Computing in marimo Notebooks

## Setup

```python
with app.setup:
    import marimo as mo
    import numpy as np
    from scipy import stats, optimize, signal, integrate, interpolate, linalg
```

Add `scipy` to PEP 723 dependencies:

```python
# /// script
# dependencies = ["marimo", "numpy", "scipy"]
# ///
```

Use namespaced imports to avoid shadowing Python's built-in `io` module (see Import Guidelines below).

## Statistical Analysis (`scipy.stats`)

### Hypothesis Tests

```python
# t-tests
t_stat, p_val = stats.ttest_ind(group1, group2)      # independent samples
t_stat, p_val = stats.ttest_rel(before, after)        # paired
t_stat, p_val = stats.ttest_1samp(data, popmean=0)    # one-sample

# Non-parametric
statistic, p_val = stats.mannwhitneyu(x, y)           # Mann-Whitney U
f_stat, p_val = stats.f_oneway(g1, g2, g3)            # one-way ANOVA

# Normality and goodness-of-fit
statistic, p_val = stats.shapiro(data)
chi2, p_val = stats.chisquare(observed, expected)
```

### Distributions

```python
from scipy.stats import norm, t, chi2, expon, binom, poisson

norm.pdf(x, loc=0, scale=1)    # PDF
norm.cdf(x)                     # CDF
norm.rvs(size=1000)             # random samples
norm.fit(data)                  # fit to data -> (mean, std)
t.ppf(0.975, df=29)            # quantile (inverse CDF)
binom.pmf(k=5, n=10, p=0.5)   # discrete PMF
```

### Correlation & Descriptive Statistics

```python
r, p = stats.pearsonr(x, y)
rho, p = stats.spearmanr(x, y)
tau, p = stats.kendalltau(x, y)
result = stats.linregress(x, y)   # slope, intercept, r, pvalue, stderr
z_scores = stats.zscore(data)
n, minmax, mean, var, skew, kurt = stats.describe(data)
ci = stats.t.interval(0.95, df=len(data)-1, loc=data.mean(), scale=stats.sem(data))
```

## Optimization (`scipy.optimize`)

```python
# Minimize multivariate function
result = optimize.minimize(f, x0, method='BFGS')
result = optimize.minimize(f, x0, method='L-BFGS-B', bounds=[(0, 1), (0, 1)])

# Minimize scalar function
result = optimize.minimize_scalar(f, bounds=(0, 5), method='bounded')

# Curve fitting (nonlinear least squares)
def model(x, a, b):
    return a * np.exp(-b * x)

popt, pcov = optimize.curve_fit(model, xdata, ydata, p0=[1.0, 1.0])

# Root finding
solution = optimize.fsolve(equations, x0)
result = optimize.root(equations, x0, method='hybr')

# Linear programming
result = optimize.linprog(c, A_ub=A, b_ub=b, bounds=(0, None))
```

## Numerical Integration (`scipy.integrate`)

```python
# Definite integral of a function
result, error = integrate.quad(f, a, b)

# Double integral
result, error = integrate.dblquad(f, a, b, gfun, hfun)

# ODE solving
sol = integrate.solve_ivp(dydt, [t0, tf], y0, t_eval=np.linspace(t0, tf, 100))
sol.y   # solution array, shape (n_states, n_timepoints)
sol.t   # time points
```

## Interpolation (`scipy.interpolate`)

```python
# 1D interpolation
f = interpolate.interp1d(x, y, kind='linear')  # 'linear', 'cubic', 'quadratic'
y_new = f(x_new)

# Cubic spline
cs = interpolate.CubicSpline(x, y)
y_new = cs(x_new)

# 2D interpolation
f2d = interpolate.RegularGridInterpolator((x_grid, y_grid), values)
```

## Signal Processing (`scipy.signal`)

```python
# Convolution
result = signal.convolve(signal1, signal2, mode='full')

# Butterworth filter design and application
b, a = signal.butter(N=4, Wn=0.3, btype='low')   # design
filtered = signal.filtfilt(b, a, data)             # apply zero-phase

# Fourier transform (prefer scipy.fft)
from scipy.fft import fft, ifft, fftfreq
freqs = fftfreq(n, d=1.0 / sample_rate)
spectrum = fft(signal_data)
```

## Linear Algebra (`scipy.linalg`)

More complete than `numpy.linalg`:

```python
linalg.inv(A)                # matrix inverse
linalg.det(A)                # determinant
linalg.solve(A, b)           # solve Ax = b (prefer over inv)
linalg.lstsq(A, b)           # least squares solution

# Decompositions
U, s, Vt = linalg.svd(A)
vals, vecs = linalg.eig(A)
L, U = linalg.lu(A)[:2]     # LU decomposition
Q, R = linalg.qr(A)         # QR decomposition
```

## Import Guidelines

- Use `from scipy import stats, optimize` rather than `import scipy.stats` for cleaner code
- **Never** `import scipy.io` — this conflicts with Python's built-in `io` module; use `import scipy; scipy.io.loadmat(...)` instead, or `from scipy import io as sio`
- SciPy uses lazy loading — only the imported submodules are actually loaded into memory
