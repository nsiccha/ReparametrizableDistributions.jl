struct R2D2{I} <: AbstractReparametrizableDistribution
    info::I
end
info(source::R2D2) = source.info
reparametrization_parameters(source::R2D2) = map(reparametrization_parameters, info(source))
reparametrize(source::R2D2, parameters::AbstractVector) = R2D2(map(
    reparametrize, info(source), views(reparametrization_parameters(source), parameters)
))
Base.length(source::R2D2) = sum(length.(values(info(source))))
# Base.getproperty(what::R2D2, key::Symbol) = hasfield(R2D2, key) ? getfield(what, key) : getproperty(what.info, key)

# log_scale_distribution(source::R2D2, ::Any) = Product([fixed(source)])
# hierarchical_distribution(source::R2D2, draw::AbstractVector) = begin 
#     log_scale = draw[1]
#     scales = exp.(log_scale .* centeredness(source))
#     Product(Normal.(0, scales))
# end
# subdistributions(source::R2D2, draw::AbstractVector) = log_scale_distribution(source, draw), hierarchical_distribution(source, draw)

# @views logdensity_and_stuff(source::R2D2, draw::AbstractVector, lpdf=0.) = begin
#     info = info(source)
#     views = views(source, draw)
#     simplex = logdensity_and_stuff(info.simplex, views.simplex)
#     sigma = sqrt.(views.sigma_sq)
#     tau = R2 ./ (1 .- R2)
#     sds = (sigma*tau) .* sqrt.(simplex[2])
#     hierarchy = logdensity_and_stuff(info.hierarchy, views.hierarchy)
#     lpdf += simplex[1]
#     lpdf += hierarchy[1]
#     lpdf, (;sds, beta)
# end

info(source::R2D2, draw::AbstractVector) = begin
    _info = info(source)
    _views = views(source, draw)
    simplex = logdensity_and_stuff(_info.simplex, _views.simplex)
    sigma = sqrt.(_views.sigma_sq)
    tau = R2 ./ (1 .- R2)
    sds = (sigma*tau) .* sqrt.(simplex[2])
    hierarchy = (;sds)
    (;_info.intercept, _info.sigma_sq, _info.R2, _info.simplex, hierarchy)
end
views(source::R2D2, draw::AbstractVector) = views(info(source), draw)

lja_reparametrize(source::R2D2, target::R2D2, draw::AbstractVector, lja=0.) = begin
    reparametrize(s, t, d) = ((lja_, draw_) = lja_reparametrize(s, t, d); lja += lja_; draw_)
    return lja, vcat(map(
        reparametrize, 
        info(source, draw), 
        info(target), 
        views(source, draw)
    )...)
    sinfo, tinfo = source.info, target.info
    sdraw = views(sinfo, draw)
    tdraw_intercept = sdraw.intercept
    tdraw_sigma_sq = sdraw.sigma_sq
    tdraw_R2 = sdraw.R2
    lja_simplex, tdraw_simplex = lja_reparametrize(sinfo.simplex, tinfo.simplex, sdraw.simplex)
    lja_hierarchy, tdraw_hierarchy = lja_reparametrize(sinfo.hierarchy, tinfo.hierarchy, sdraw.hierarchy)
    lja += lja_simplex + lja_hierarchy
    tdraw = vcat(tdraw_intercept, tdraw_sigma_sq, tdraw_R2, tdraw_simplex, tdraw_hierarchy)
    lja, tdraw
end