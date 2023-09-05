struct R2D2{I} <: AbstractReparametrizableDistribution
    info::I
end
R2D2(log_sigma, R2, simplex, hierarchy) = R2D2((;log_sigma, R2, simplex, hierarchy))
reparametrize(source::R2D2, parameters::NamedTuple) = R2D2(
    map(reparametrize, info(source), parameters)...
)

logdensity_and_stuff(source::R2D2, draw::AbstractVector, lpdf=0.) = begin
    _info, _views = info_and_views(source, draw)
    sigma = exp.(_views.log_sigma)
    tau = _views.R2 ./ (1 .- _views.R2)
    simplex = logdensity_and_stuff(_info.simplex, _views.simplex)
    scales = (sigma.*tau) .* sqrt.(simplex.weights)
    prior_hierarchy = Normal(0, scales .^ _centeredness)
    hierarchy_lpdf = sum_logpdf(prior_hierarchy, _views.hierarchy)
    hierarchy = scales .^ (1 .- _centeredness) .* _views.hierarchy
    lpdf += simplex.lpdf
    lpdf += hierarchy_lpdf
    (;lpdf, simplex, sigma, tau, scales, hierarchy)
end

lja_reparametrize(source::R2D2, target::R2D2, draw::AbstractVector, lja=0.) = begin
    _info, _views = info_and_views(source, draw)
    _centeredness = info(_info.hierarchy).centeredness
    sigma = exp.(_views.log_sigma)
    tau = exp.(_views.R2)# ./ (1 .- _views.R2)
    simplex = logdensity_and_stuff(_info.simplex, _views.simplex)
    scales = (sigma .* tau) .* sqrt.(simplex.weights)

    tinfo = info(target)
    tcenteredness = info(tinfo.hierarchy).centeredness
    lja_simplex, tdraw_simplex = lja_reparametrize(_info.simplex, tinfo.simplex, _views.simplex)
    tprior_hierarchy = Normal.(0., scales .^ tcenteredness)
    tdraw_hierarchy = _views.hierarchy .* scales .^ (tcenteredness .- _centeredness)
    lja_hierarchy = sum_logpdf(tprior_hierarchy, tdraw_hierarchy)
    lja += lja_simplex 
    lja += lja_hierarchy
    tdraw = vcat(_views.log_sigma, _views.R2, tdraw_simplex, tdraw_hierarchy)
    lja, tdraw
end