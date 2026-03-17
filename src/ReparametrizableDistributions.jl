module ReparametrizableDistributions

using Distributions, LinearAlgebra, LogExpFunctions

export FusedTransform, advance!!, fused_logdensity, constrain
export NormalTransform, LKJCholeskyTransform, AffineNormalTransform
export CorrelatedEffectsTransform, scales_from_cholesky

"""
    advance!!(x, pos)

Consume one element from `x` at position `pos+1`. Returns `(value, new_pos)`.
"""
advance!!(x, pos) = x[pos+1], pos+1

"""
    advance!!(x, pos, n)

Consume `n` elements from `x` starting at `pos+1`. Returns `(view, new_pos)`.
"""
advance!!(x, pos, n) = view(x, pos+1:pos+n), pos+n

"""
    FusedTransform

Abstract supertype for fused bijector + log-prior-density transforms.

A `FusedTransform` jointly:
1. Maps unconstrained parameters to constrained space (the bijection)
2. Computes the log prior density contribution (including the Jacobian adjustment)

This avoids the overhead and numerical issues of computing the transformation
and its log-Jacobian-determinant separately.

Subtypes must implement [`fused_logdensity`](@ref).
"""
abstract type FusedTransform end

"""
    fused_logdensity(t::FusedTransform, x; init=(0.0, 0))

Consume unconstrained parameters from `x`, write the constrained result into `t`,
and return `(logdensity, pos)` where `logdensity` is the accumulated log prior density
(including Jacobian adjustments) and `pos` is the new position in `x`.

`init` is a tuple `(logdensity_accumulator, starting_position)`.
"""
function fused_logdensity end

include("log_abs_tanh.jl")
include("transforms/normal.jl")
include("transforms/affine_normal.jl")
include("transforms/lkj_cholesky.jl")
include("transforms/correlated_effects.jl")

end # module ReparametrizableDistributions
