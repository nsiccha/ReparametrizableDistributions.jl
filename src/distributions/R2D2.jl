struct R2D2{I} <: AbstractReparametrizableDistribution
    info::I
end
R2D2(log_sigma, logit_R2, simplex, hierarchy) = R2D2((;log_sigma, logit_R2, simplex, hierarchy))
reparametrize(source::R2D2, parameters::NamedTuple) = R2D2(map(reparametrize, info(source), parameters))

lpdf_and_invariants(source::R2D2, draw::NamedTuple, lpdf=0.) = begin
    _info = info(source)
    sigma = exp.(draw.log_sigma)
    R2 = logistic.(draw.logit_R2)
    tau = R2 ./ (1 .- R2)
    simplex = lpdf_and_invariants(_info.simplex, draw.simplex)
    log_scale = log.((sigma.*tau) .* sqrt.(simplex.weights))
    hierarchy = lpdf_and_invariants(_info.hierarchy, (;log_scale, xic=draw.hierarchy))
    lpdf += sum_logpdf(_info.log_sigma, draw.log_sigma)
    lpdf += sum_logpdf(_info.logit_R2, draw.logit_R2)
    lpdf += simplex.lpdf
    lpdf += hierarchy.lpdf
    (;lpdf, draw.log_sigma, draw.logit_R2, sigma, R2, tau, simplex, hierarchy)
end