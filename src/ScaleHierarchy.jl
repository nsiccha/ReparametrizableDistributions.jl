struct ScaleHierarchy{F,V} <: AbstractReparametrizableDistribution
    fixed::F
    variable::V
end
WarmupHMC.reparametrization_parameters(source::ScaleHierarchy) = variable(source)
WarmupHMC.reparametrize(source::ScaleHierarchy, parameters::AbstractVector) = ScaleHierarchy(
    fixed(source), parameters
)

centeredness(source::ScaleHierarchy) = variable(source)
Base.length(source::ScaleHierarchy) = 1+length(centeredness(source))

log_scale_distribution(source::ScaleHierarchy, ::Any) = Product([fixed(source)])
hierarchical_distribution(source::ScaleHierarchy, draw::AbstractVector) = begin 
    log_scale = draw[1]
    scales = exp.(log_scale .* centeredness(source))
    Product(Normal.(0, scales))
end
subdistributions(source::ScaleHierarchy, draw::AbstractVector) = log_scale_distribution(source, draw), hierarchical_distribution(source, draw)

@views WarmupHMC.logdensity_and_stuff(source::ScaleHierarchy, draw::AbstractVector, lpdf=0.) = begin 
    sdists = subdistributions(source, draw)
    log_scale, xic = subdraws = views(sdists, draw)
    lpdf += sum(logpdf.(sdists, subdraws))
    x = xic .* exp.(log_scale .* (1 .- centeredness(source)))
    lpdf, x
end

WarmupHMC.lja_reparametrize(source::ScaleHierarchy, target::ScaleHierarchy, draw::AbstractVector, lja=0.) = begin 
    log_scale, sxic = views(subdistributions(source, draw), draw)
    txic = sxic .* exp.(log_scale .* (centeredness(target) .- centeredness(source)))
    tdraw = vcat(log_scale, txic)
    lja += logpdf(hierarchical_distribution(target, tdraw), txic)
    lja, tdraw
end