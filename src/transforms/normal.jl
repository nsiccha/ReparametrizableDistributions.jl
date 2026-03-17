"""
    NormalTransform(n::Int)

Fused identity transform + standard normal prior for `n` unconstrained parameters.

Consumes `n` values from the unconstrained vector and accumulates
`sum(logpdf(Normal(), xi))` into the log density.
"""
struct NormalTransform <: FusedTransform
    n::Int
end

function fused_logdensity(t::NormalTransform, x; init=(0.0, 0))
    lp, pos = init
    xi, pos = advance!!(x, pos, t.n)
    lp += sum(Base.Fix1(logpdf, Normal()), xi)
    lp, pos
end
