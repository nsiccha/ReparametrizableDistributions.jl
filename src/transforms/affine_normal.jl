"""
    AffineNormalTransform(offset, multiplier; centered=0.0)

Fused affine transform + normal prior with partial centering, following
Gorinova, Moore & Hoffman (2019) "Automatic Reparameterisation of Probabilistic Programs".

Given unconstrained parameters `z`, produces constrained parameters
`x = offset + multiplier^(1-c) * (z - c * offset)` where `c` is the centering
parameter and `z ~ Normal(c * offset, multiplier^c)`.

**Centering parameter `c ∈ [0, 1]`:**
- `c = 0` (default): Fully non-centered. `z ~ Normal(0, 1)`, `x = offset + multiplier * z`.
  Best when data is uninformative relative to the prior.
- `c = 1`: Fully centered. `z ~ Normal(offset, multiplier)`, `x = z`.
  Best when data strongly constrains the parameter.
- `0 < c < 1`: Partial centering. Interpolates between the two extremes.
  Can be optimized to find the best geometry for sampling.

**Arguments:**
- `offset`: Location parameter (scalar or vector). Corresponds to Stan's `offset`.
- `multiplier`: Scale parameter (positive, scalar or vector). Corresponds to Stan's `multiplier`.
- `centered`: Centering parameter(s), scalar or vector in `[0, 1]`.

The number of unconstrained parameters consumed equals `length(offset)` (or 1 if scalar).
"""
struct AffineNormalTransform{O,M,C} <: FusedTransform
    offset::O
    multiplier::M
    centered::C
end

AffineNormalTransform(offset, multiplier; centered=0.0) =
    AffineNormalTransform(offset, multiplier, centered)

nparams(t::AffineNormalTransform{<:Real}) = 1
nparams(t::AffineNormalTransform{<:AbstractVector}) = length(t.offset)

function fused_logdensity(t::AffineNormalTransform{<:Real}, x; init=(0.0, 0))
    (; offset, multiplier, centered) = t
    lp, pos = init
    z, pos = advance!!(x, pos)
    # Prior: z ~ Normal(centered * offset, multiplier^centered)
    z_loc = centered * offset
    z_scale = multiplier^centered
    lp += logpdf(Normal(z_loc, z_scale), z)
    lp, pos
end

function fused_logdensity(t::AffineNormalTransform{<:AbstractVector}, x; init=(0.0, 0))
    (; offset, multiplier, centered) = t
    n = length(offset)
    lp, pos = init
    zi, pos = advance!!(x, pos, n)
    # Prior: z_j ~ Normal(centered_j * offset_j, multiplier_j^centered_j)
    for j in 1:n
        c_j = centered isa Real ? centered : centered[j]
        z_loc = c_j * offset[j]
        z_scale = multiplier[j]^c_j
        lp += logpdf(Normal(z_loc, z_scale), zi[j])
    end
    lp, pos
end

"""
    constrain(t::AffineNormalTransform, z)

Recover the constrained parameter `x` from unconstrained `z`:
`x = offset + multiplier^(1-c) * (z - c * offset)`.
"""
constrain(t::AffineNormalTransform{<:Real}, z::Real) = begin
    (; offset, multiplier, centered) = t
    offset + multiplier^(1 - centered) * (z - centered * offset)
end

constrain(t::AffineNormalTransform{<:AbstractVector}, z::AbstractVector) = begin
    (; offset, multiplier, centered) = t
    map(eachindex(offset)) do j
        c_j = centered isa Real ? centered : centered[j]
        offset[j] + multiplier[j]^(1 - c_j) * (z[j] - c_j * offset[j])
    end
end
