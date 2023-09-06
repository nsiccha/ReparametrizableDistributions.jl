struct R2D2{I} <: AbstractReparametrizableDistribution
    info::I
end
R2D2(log_sigma, R2, simplex, hierarchy) = R2D2((;log_sigma, R2, simplex, hierarchy))
reparametrize(source::R2D2, parameters::NamedTuple) = R2D2(
    map(reparametrize, info(source), parameters)...
)

lpdf_and_invariants(source::R2D2, draw::NamedTuple, lpdf=0.) = begin
    _info = info(source)
    # _centeredness = info(_info.hierarchy).centeredness
    sigma = exp.(draw.log_sigma)
    tau = exp.(draw.R2)# ./ (1 .- draw.R2)
    simplex = lpdf_and_invariants(_info.simplex, draw.simplex)
    log_scale = log.((sigma.*tau) .* sqrt.(simplex.weights))
    hierarchy = lpdf_and_invariants(_info.hierarchy, (;log_scale, xic=draw.hierarchy))
    # prior_hierarchy = Normal.(0, scales .^ _centeredness)
    # hierarchy_lpdf = sum_logpdf(prior_hierarchy, draw.hierarchy)
    # hierarchy = scales .^ (1 .- _centeredness) .* draw.hierarchy
    lpdf += simplex.lpdf
    lpdf += hierarchy.lpdf
    (;lpdf, draw.log_sigma, draw.R2, sigma, tau, simplex, hierarchy)
end

lja_reparametrize(source::R2D2, target::R2D2, invariants::NamedTuple, lja=0.) = begin
    _info = info(source)
    # _centeredness = info(_info.hierarchy).centeredness
    tinfo = info(target)
    # tcenteredness = info(tinfo.hierarchy).centeredness

    lja_simplex, tdraw_simplex = lja_reparametrize(_info.simplex, tinfo.simplex, invariants.simplex)
    lja_hierarchy, tdraw_hierarchy = lja_reparametrize(_info.hierarchy, tinfo.hierarchy, invariants.hierarchy)
    # tprior_hierarchy = Normal.(0., invariants.scales .^ tcenteredness)
    # tdraw_hierarchy = invariants.hierarchy .* invariants.scales .^ (tcenteredness .- 1)
    # lja_hierarchy = sum_logpdf(tprior_hierarchy, tdraw_hierarchy)
    lja += lja_simplex 
    lja += lja_hierarchy
    tdraw = vcat(invariants.log_sigma, invariants.R2, tdraw_simplex, views(tdraw_hierarchy).xic)
    lja, tdraw
end