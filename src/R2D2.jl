struct R2D2{I} <: AbstractReparametrizableDistribution
    info::I
end
R2D2(log_sigma, R2, simplex, hierarchy) = R2D2((;log_sigma, R2, simplex, hierarchy))
reparametrize(source::R2D2, parameters::NamedTuple) = R2D2(
    map(reparametrize, info(source), parameters)...
)

lpdf_and_invariants(source::R2D2, draw::NamedTuple, lpdf=0.) = begin
    _info = info(source)
    sigma = exp.(draw.log_sigma)
    tau = draw.R2 ./ (1 .- draw.R2)
    simplex = lpdf_and_invariants(_info.simplex, draw.simplex)
    scales = (sigma.*tau) .* sqrt.(simplex.weights)
    prior_hierarchy = Normal(0, scales .^ _centeredness)
    hierarchy_lpdf = sum_logpdf(prior_hierarchy, draw.hierarchy)
    hierarchy = scales .^ (1 .- _centeredness) .* draw.hierarchy
    lpdf += simplex.lpdf
    lpdf += hierarchy_lpdf
    (;lpdf, draw.log_sigma, draw.R2, sigma, tau, simplex, scales, hierarchy)
end

lja_reparametrize(source::R2D2, target::R2D2, draw::NamedTuple, lja=0.) = begin
    _info = info(source)
    _centeredness = info(_info.hierarchy).centeredness
    sigma = exp.(draw.log_sigma)
    tau = exp.(draw.R2)# ./ (1 .- draw.R2)
    simplex = lpdf_and_invariants(_info.simplex, draw.simplex)
    scales = (sigma .* tau) .* sqrt.(simplex.weights)

    tinfo = info(target)
    tcenteredness = info(tinfo.hierarchy).centeredness
    lja_simplex, tdraw_simplex = lja_reparametrize(_info.simplex, tinfo.simplex, draw.simplex)
    tprior_hierarchy = Normal.(0., scales .^ tcenteredness)
    tdraw_hierarchy = draw.hierarchy .* scales .^ (tcenteredness .- _centeredness)
    lja_hierarchy = sum_logpdf(tprior_hierarchy, tdraw_hierarchy)
    lja += lja_simplex 
    lja += lja_hierarchy
    tdraw = vcat(draw.log_sigma, draw.R2, tdraw_simplex, tdraw_hierarchy)
    lja, tdraw
end