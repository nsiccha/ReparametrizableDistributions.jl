# ReparametrizableDistributions.jl

Fused bijector + log-prior-density transforms for reparametrizing distributions/posteriors to make them easier to sample from using MCMC methods.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/nsiccha/ReparametrizableDistributions.jl")
```

## Usage

```julia
using ReparametrizableDistributions

# Normal transform: unconstrained → Normal(0, 1)
t = NormalTransform(0.0)
logd, pos = fused_logdensity(t, [0.5])

# LKJ Cholesky transform for correlation matrices
t = LKJCholeskyTransform(3, 2.0)
np = nparams(t)  # number of unconstrained parameters
```

## Transforms

- `NormalTransform` — standard normal prior
- `AffineNormalTransform` — partially centered normal (Gorinova et al. 2019)
- `LKJCholeskyTransform` — LKJ Cholesky correlation matrix
- `CorrelatedEffectsTransform` — correlated random effects with partial centering
