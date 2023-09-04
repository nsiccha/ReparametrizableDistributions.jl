struct R2D2{I} <: AbstractReparametrizableDistribution
    info::I
end
info(source::R2D2) = source.info
Base.length(source::R2D2) = sum(length.(values(info(source))))
views(source::R2D2, draw::AbstractVector) = views(info(source), draw)
reparametrization_parameters(source::R2D2) = map(reparametrization_parameters, info(source))
reparametrize(source::R2D2, parameters::AbstractVector) = R2D2(map(
    reparametrize, info(source), views(reparametrization_parameters(source), parameters)
))

@views logdensity_and_stuff(source::R2D2, draw::AbstractVector, lpdf=0.) = begin
    _info = info(source)
    _views = views(source, draw)
    simplex = logdensity_and_stuff(_info.simplex, _views.simplex)
    sigma = sqrt.(_views.sigma_sq)
    tau = _views.R2 ./ (1 .- _views.R2)
    scales = (sigma*tau) .* sqrt.(simplex)
    hierarchy_lpdf = sum(logpdf.(Normal(0, scales .^ _info.hierarchy)))
    hierarchy = scales .^ (1 .- _info.hierarchy) .* _views.hierarchy
    lpdf += simplex_lpdf
    lpdf += hierarchy_lpdf
    lpdf, (;scales, hierarchy)
end

lja_reparametrize(source::R2D2, target::R2D2, draw::AbstractVector, lja=0.) = begin
    sinfo, tinfo = info.((source, target))
    sdraw = views(source, draw)
    tdraw_intercept = sdraw.intercept
    tdraw_sigma_sq = sdraw.sigma_sq
    tdraw_R2 = sdraw.R2
    lja_simplex, tdraw_simplex = lja_reparametrize(sinfo.simplex, tinfo.simplex, sdraw.simplex)
    lja_hierarchy = sum(logpdf.(Normal(0, sds .^ _info.hierarchy)))
    lja += lja_simplex + lja_hierarchy
    tdraw = vcat(tdraw_intercept, tdraw_sigma_sq, tdraw_R2, tdraw_simplex, tdraw_hierarchy)
    lja, tdraw
end