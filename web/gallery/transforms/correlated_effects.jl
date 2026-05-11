# title: CorrelatedEffectsTransform
# description: Partially-centered correlated random effects θ = L · diag(s^{-c}) · z. Wraps an LKJ-Cholesky factor and shares a centering parameter c ∈ [0, 1] across dims.
# section: transforms
# id: correlated_effects
# tags: random_effects, correlation, centering

let lkj      = LKJCholeskyTransform(3; eta=2.0),
    np_lkj   = nparams(lkj),
    _, _     = fused_logdensity(lkj, randn(np_lkj)),
    L        = lkj.L,
    scales   = scales_from_cholesky(L, 3),
    effects  = zeros(5, 3)
    CorrelatedEffectsTransform(L, scales, effects; centered=0.5)
end
