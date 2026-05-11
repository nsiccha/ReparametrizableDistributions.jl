```@meta
CurrentModule = ReparametrizableDistributions
```

# ReparametrizableDistributions.jl

Fused bijector + log-prior-density transforms for reparametrizing
distributions/posteriors to make them easier to sample from using MCMC
methods.

A [`FusedTransform`](@ref) jointly maps unconstrained parameters to
constrained space *and* accumulates the log prior density (including
Jacobian adjustments) in a single pass. This avoids the overhead and
numerical issues of computing the transformation and its
log-Jacobian-determinant separately.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/nsiccha/ReparametrizableDistributions.jl")
```

## Quick start

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

## Available transforms

| Transform | Constrained space | Notes |
|---|---|---|
| [`NormalTransform`](@ref) | `ℝⁿ` (identity) | Standard normal prior |
| [`AffineNormalTransform`](@ref) | `ℝⁿ` (affine) | Partial centering, Gorinova et al. 2019 |
| [`LKJCholeskyTransform`](@ref) | Cholesky factor `L` of an `n×n` correlation matrix | LKJ(η) prior |
| [`CorrelatedEffectsTransform`](@ref) | `n_groups × n_dims` random effects | Partial-centering wrapper around an LKJ-Cholesky factor |

See the [Gallery](gallery.md) for live demos of each transform, and the
[API Reference](api.md) for full docstrings.

## See also

- [WarmupHMC.jl](https://github.com/nsiccha/WarmupHMC.jl) — adaptive HMC
  warm-up that picks the centering parameter for these transforms.
- [StanBlocks.jl](https://github.com/nsiccha/StanBlocks.jl) — Stan-model DSL
  that uses related reparametrizations.
