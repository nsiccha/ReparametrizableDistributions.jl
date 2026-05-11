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

# Standard normal prior on n parameters
t = NormalTransform(3)
x = randn(nparams(t))
logd, pos = fused_logdensity(t, x)

# LKJ Cholesky transform for a 3×3 correlation matrix
t = LKJCholeskyTransform(3; eta=2.0)
x = randn(nparams(t))
logd, pos = fused_logdensity(t, x)
# `t.L` now holds the constrained Cholesky factor
```

## Transforms

- `NormalTransform` — standard normal prior
- `AffineNormalTransform` — partially centered normal (Gorinova et al. 2019)
- `LKJCholeskyTransform` — LKJ Cholesky correlation matrix
- `CorrelatedEffectsTransform` — correlated random effects with partial centering
