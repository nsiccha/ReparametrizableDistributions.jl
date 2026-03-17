"""
    CorrelatedEffectsTransform(L, scales, effects; centered=0.0)

Fused partially-centered transform for correlated random effects.

Given a Cholesky factor `L` (from [`LKJCholeskyTransform`](@ref)) with row scales
`scales`, this transform consumes `n_groups × n_dims` unconstrained parameters and
produces correlated random effects `θ = L · diag(s^{-c}) · z`.

**Parametrization:**
- `z_i ~ N(0, s_i^{c_i})` independently (factored prior, each depends on one `c_i`)
- `η_i = z_i / s_i^{c_i} ~ N(0, 1)` (standardized)
- `θ = L · η` (Cholesky mixing, no `c` dependency)

This is exact: `θ ~ N(0, L·L')` for any choice of `c`.

**Centering parameter `c ∈ [0, 1]`:**
- `c = 0`: Fully non-centered. `z ~ N(0, I)`, `θ = L · z`. Prior on `z` is independent
  of scale hyperparameters.
- `c = 1`: `z_i ~ N(0, s_i)`. Prior on `z` depends on scales but not correlations.
  Not identical to the traditional centered parametrization (which requires a correlated
  prior), but gives the correct distribution on `θ`.
- `0 < c < 1`: Partial centering interpolating between the extremes.

**Arguments:**
- `L`: Lower-triangular Cholesky factor (mutated by `LKJCholeskyTransform`)
- `scales`: Vector of row scales `s_i = exp(log_scale_i)` from the Cholesky construction
- `effects`: Matrix of size `(n_groups, n_dims)` — mutable storage for the constrained effects
- `centered`: Centering parameter(s), scalar or vector of length `n_dims` in `[0, 1]`
"""
struct CorrelatedEffectsTransform{TL,TS,TE,TC} <: FusedTransform
    L::TL
    scales::TS
    effects::TE
    centered::TC
end

CorrelatedEffectsTransform(L, scales, effects; centered=0.0) =
    CorrelatedEffectsTransform(L, scales, effects, centered)

nparams(t::CorrelatedEffectsTransform) = length(t.effects)

function fused_logdensity(t::CorrelatedEffectsTransform, x; init=(0.0, 0))
    (; L, scales, effects, centered) = t
    lp, pos = init
    n_dims = length(scales)
    n_groups = size(effects, 1)
    for j in 1:n_groups
        zi, pos = advance!!(x, pos, n_dims)
        # Accumulate factored prior: z_i ~ N(0, s_i^{c_i})
        for i in 1:n_dims
            c_i = centered isa Real ? centered : centered[i]
            s_i = scales[i]
            lp += logpdf(Normal(0, s_i^c_i), zi[i])
        end
        # Constrain: η = z ./ s.^c, then θ_j = L · η
        # effects[j, :] = L * (z ./ s.^c)
        for i in 1:n_dims
            c_i = centered isa Real ? centered : centered[i]
            acc = zero(eltype(x))
            for k in 1:i
                c_k = centered isa Real ? centered : centered[k]
                acc += L[i, k] * zi[k] / scales[k]^c_k
            end
            effects[j, i] = acc
        end
    end
    lp, pos
end

"""
    scales_from_cholesky(L, n)

Extract row scales from a Cholesky factor `L` produced by [`LKJCholeskyTransform`](@ref).

In the LKJ Cholesky parametrization, `L[i,:] = s_i * L_corr[i,:]` where `s_i` is the
row scale. Since `L_corr` has the property that `sum(L_corr[i,k]^2 for k=1:i) = 1`,
the scale is `s_i = norm(L[i, 1:i])`.
"""
function scales_from_cholesky(L, n)
    [sqrt(sum(L[i, k]^2 for k in 1:i)) for i in 1:n]
end
