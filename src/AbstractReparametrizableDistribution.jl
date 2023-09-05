abstract type AbstractReparametrizableDistribution <: ContinuousMultivariateDistribution end

# fixed(source::AbstractReparametrizableDistribution) = source.fixed
# variable(source::AbstractReparametrizableDistribution) = source.variable
LogDensityProblems.dimension(source::AbstractReparametrizableDistribution) = length(source)
LogDensityProblems.logdensity(source::AbstractReparametrizableDistribution, draw::AbstractVector) = WarmupHMC.logdensity_and_stuff(source, draw).lpdf

default_info(::AbstractReparametrizableDistribution) = NamedTuple()
info(source::AbstractReparametrizableDistribution) = merge(default_info(source), source.info)
info_and_views(source::AbstractReparametrizableDistribution, draw::AbstractVector) = info(source), views(source, draw)
Base.length(source::AbstractReparametrizableDistribution) = sum(length.(values(info(source))))
views(source::AbstractReparametrizableDistribution, draw::AbstractVector) = views(info(source), draw)
reparametrization_parameters(source::AbstractReparametrizableDistribution) = vcat(
    map(reparametrization_parameters, info(source))...
)
reparametrize(source::AbstractReparametrizableDistribution, parameters::AbstractVector) = reparametrize(
    source, views(map(reparametrization_parameters, info(source)), parameters)
)
# Base.isapprox(lhs::AbstractReparametrizableDistribution, rhs::AbstractReparametrizableDistribution) = begin 
#     linfo, rinfo = info.((lhs, rhs))
#     keys(linfo) == keys(rinfo) && all(map(isapprox, linfo, rinfo))
# end
# Base.length(source::AbstractReparametrizableDistribution) = sum(length.(info(source)))
# views(source::AbstractReparametrizableDistribution, draw::AbstractVector) = views(info(source), draw)
# reparametrization_parameters(source::AbstractReparametrizableDistribution) = map(reparametrization_parameters, info(source))
# reparametrize(source::R2D2, parameters::AbstractVector) = R2D2(map(
#     reparametrize, info(source), views(reparametrization_parameters(source), parameters)
# ))