struct R2D2{I} <: AbstractReparametrizableDistribution
    info::I
end
reparametrization_parameters(source::R2D2) = map(reparametrization_parameters, source.info)
reparametrize(source::R2D2, parameters::AbstractVector) = R2D2(map(
    reparametrize, source.info, views(reparametrization_parameters(source), parameters)
))
Base.length(source::R2D2) = sum(length.(values(source.info)))
# Base.getproperty(what::R2D2, key::Symbol) = hasfield(R2D2, key) ? getfield(what, key) : getproperty(what.info, key)

# log_scale_distribution(source::R2D2, ::Any) = Product([fixed(source)])
# hierarchical_distribution(source::R2D2, draw::AbstractVector) = begin 
#     log_scale = draw[1]
#     scales = exp.(log_scale .* centeredness(source))
#     Product(Normal.(0, scales))
# end
# subdistributions(source::R2D2, draw::AbstractVector) = log_scale_distribution(source, draw), hierarchical_distribution(source, draw)

@views logdensity_and_stuff(source::R2D2, draw::AbstractVector, lpdf=0.) = begin
    intercept, sigma_sq, R2 = draw[1:3]
    simplex_draw, hierarchical_draw = reshape(draw[4:end], (:, 2))
    simplex_lpdf, simplex_x = logdensity_and_stuff(simplex(source), simplex_draw)
    lpdf += simplex_lpdf
    sigma = sqrt(sigma_sq)
    tau = R2 / (1 - R2)
    sds = (sigma*tau) .* sqrt.(simplex_x)
    lpdf += sum(logpdf.(Normal(0, sds .^ centeredness(source))))
    beta = hierarchical_draw .* sds .^ (1 - centeredness(source))
    # hierarchical_lpdf, hierarchical_draw = logdensity_and_stuff
    # lpdf += hierarchical_lpdf
    lpdf, beta
end

lja_reparametrize(source::R2D2, target::R2D2, draw::AbstractVector, lja=0.) = begin
    reparametrize(s, t, d) = ((lja_, draw_) = lja_reparametrize(s, t, d); lja += lja_; draw_)
    return lja, vcat([
        reparametrize.(distributions(source, draw), distributions(target), draws(source, draw))
    ]...)
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