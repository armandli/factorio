# PyTorch Autograd, torch.func, and Distributions Reference

## torch.autograd

### Basic Gradient Computation

```python
x = torch.randn(3, requires_grad=True)
y = (x ** 2).sum()
y.backward()      # accumulates gradient into x.grad
print(x.grad)     # tensor of same shape as x

# Gradient of non-scalar output: pass gradient vector
y = x ** 2
y.backward(torch.ones_like(y))   # dy/dx for each element
```

### Gradient Context Managers

```python
# Disable gradient tracking (inference, eval)
with torch.no_grad():
    out = model(x)

# Enable gradient tracking inside no_grad region
with torch.no_grad():
    with torch.enable_grad():
        y = model(x)   # tracked

# Set globally
torch.set_grad_enabled(False)
torch.set_grad_enabled(True)

# Functional version (decorator)
@torch.no_grad()
def predict(model, x):
    return model(x)
```

### torch.autograd.grad

```python
# Compute gradient without calling .backward()
grads = torch.autograd.grad(
    outputs=loss,
    inputs=model.parameters(),
    create_graph=False,     # True: allows higher-order gradients
    retain_graph=False,     # True: keep graph for multiple backward calls
    allow_unused=False,
)

# Gradient of outputs wrt inputs (not model params)
grad_x = torch.autograd.grad(outputs=y, inputs=x)[0]
```

### Custom Autograd Function

```python
class MySigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        result = torch.sigmoid(x)
        ctx.save_for_backward(result)   # tensors to reuse in backward
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return grad_output * result * (1 - result)

# Use via .apply()
y = MySigmoid.apply(x)
```

### Debugging

```python
# Detect NaN in gradients (prints traceback to forward op that caused it)
with torch.autograd.detect_anomaly():
    loss.backward()

# Verify custom backward with finite differences
torch.autograd.gradcheck(MySigmoid.apply, inputs=(x.double(),))
torch.autograd.gradgradcheck(MySigmoid.apply, inputs=(x.double(),))
```

---

## torch.func (Functional API / functorch)

`torch.func` provides JAX-style composable transforms over stateless functions.

```python
import torch
from torch import func
```

### grad

```python
# Gradient of a scalar-returning function wrt first arg
grad_fn = func.grad(loss_fn)
grad = grad_fn(params, x, y)

# Gradient wrt multiple args
grad_fn = func.grad(loss_fn, argnums=(0, 1))
grad_params, grad_x = grad_fn(params, x, y)

# Value and gradient
val_grad_fn = func.grad_and_value(loss_fn)
grad, loss = val_grad_fn(params, x, y)
```

### vmap (vectorized map)

```python
# Vectorize a function over a batch dimension
batched_fn = func.vmap(single_sample_fn)
results = batched_fn(batch_x)   # maps over first dim by default

# Control which arg is batched (in_dims) and result stacking (out_dims)
batched_fn = func.vmap(fn, in_dims=(0, None), out_dims=0)
# in_dims=None means "broadcast" (not batched)

# Per-sample gradients
per_sample_grad = func.vmap(func.grad(loss_per_sample))
grads = per_sample_grad(params, x, y)   # shape: (batch, ...)
```

### jacrev / jacfwd (Jacobians)

```python
# Jacobian via reverse-mode (efficient when out_dim << in_dim)
jac_fn = func.jacrev(f, argnums=0)
J = jac_fn(x)   # shape: (*out_shape, *in_shape)

# Jacobian via forward-mode (efficient when in_dim << out_dim)
jac_fn = func.jacfwd(f, argnums=0)
J = jac_fn(x)

# Hessian = jacrev(jacrev(f))
hess_fn = func.jacrev(func.jacrev(f))
H = hess_fn(x)

# Hessian-vector product (efficient)
hvp = func.jvp(func.grad(f), (x,), (v,))
```

### jvp / vjp (Jacobian products)

```python
# Forward-mode: Jacobian-vector product
primals_out, tangents_out = func.jvp(f, primals=(x,), tangents=(v,))

# Reverse-mode: vector-Jacobian product
primals_out, vjp_fn = func.vjp(f, x)
grads = vjp_fn(cotangent)
```

### Functional model calls (stateless)

```python
from torch.func import functional_call

# Call a model with explicit parameter dict (no in-place state mutation)
params = dict(model.named_parameters())
buffers = dict(model.named_buffers())
output = functional_call(model, {**params, **buffers}, (x,))

# Per-sample gradients for an nn.Module
def loss_per_sample(params, buffers, x, y):
    out = functional_call(model, {**params, **buffers}, (x.unsqueeze(0),))
    return nn.CrossEntropyLoss()(out, y.unsqueeze(0))

per_sample_grads = func.vmap(func.grad(loss_per_sample), in_dims=(None, None, 0, 0))
grads = per_sample_grads(params, buffers, batch_x, batch_y)
```

---

## torch.distributions

```python
from torch import distributions as dist

# Discrete
d = dist.Bernoulli(probs=torch.tensor(0.3))
d = dist.Categorical(probs=torch.tensor([0.1, 0.4, 0.5]))
d = dist.Binomial(total_count=10, probs=torch.tensor(0.3))
d = dist.Geometric(probs=torch.tensor(0.3))

# Continuous
d = dist.Normal(loc=0.0, scale=1.0)
d = dist.MultivariateNormal(loc=mu, covariance_matrix=sigma)
d = dist.Beta(concentration1=0.5, concentration0=0.5)
d = dist.Gamma(concentration=1.0, rate=1.0)
d = dist.Exponential(rate=1.0)
d = dist.Laplace(loc=0.0, scale=1.0)
d = dist.LogNormal(loc=0.0, scale=1.0)
d = dist.Cauchy(loc=0.0, scale=1.0)
d = dist.Uniform(low=0.0, high=1.0)
d = dist.Gumbel(loc=0.0, scale=1.0)
d = dist.Dirichlet(concentration=torch.ones(3))

# Key methods (all distributions)
x = d.sample()                  # non-differentiable sample
x = d.rsample()                 # reparameterized sample (differentiable)
log_p = d.log_prob(x)          # log density/mass at x
p = d.entropy()                # entropy
cdf = d.cdf(x)                 # cumulative distribution function
icdf = d.icdf(p)               # inverse CDF (quantile)
mean = d.mean
var = d.variance
std = d.stddev
mode = d.mode

# KL divergence
kl = dist.kl_divergence(p, q)   # E_p[log p(x) - log q(x)]

# Transforms
from torch.distributions import transforms
d = dist.TransformedDistribution(
    dist.Normal(0, 1),
    [transforms.ExpTransform()]   # log-normal via transformation
)
```

### VAE Example using rsample

```python
class VAE(nn.Module):
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        q = dist.Normal(mu, std)
        return q.rsample()   # differentiable sample

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        # KL from standard normal
        kl = dist.kl_divergence(dist.Normal(mu, torch.exp(0.5*logvar)),
                                 dist.Normal(0, 1)).sum(-1).mean()
        return recon, kl
```
