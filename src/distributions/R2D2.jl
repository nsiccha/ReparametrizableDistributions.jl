struct R2D2{I} <: AbstractCompositeReparametrizableDistribution
    info::I
end
R2D2(log_sigma, logit_R2, simplex, hierarchy) = R2D2((;log_sigma, logit_R2, simplex, hierarchy))
parts(source::R2D2) = source.info
# reparametrize(source::R2D2, parameters::NamedTuple) = R2D2(map(reparametrize, info(source), parameters))

lpdf_update(source::R2D2, draw::NamedTuple, lpdf=0.) = begin
    sigma = exp.(draw.log_sigma)
    R2 = logistic.(draw.logit_R2)
    tau = R2 ./ (1 .- R2)
    simplex = lpdf_and_invariants(source.simplex, draw.simplex)
    log_scale = log.((sigma.*tau) .* sqrt.(1e-16 .+ simplex.weights))
    hierarchy = lpdf_and_invariants(source.hierarchy, (;log_scale, weights=draw.hierarchy))
    lpdf += sum_logpdf(source.log_sigma, draw.log_sigma)
    lpdf += sum_logpdf(source.logit_R2, draw.logit_R2)
    lpdf += simplex.lpdf
    lpdf += hierarchy.lpdf
    (;lpdf, simplex, hierarchy, sigma, R2, tau)
end