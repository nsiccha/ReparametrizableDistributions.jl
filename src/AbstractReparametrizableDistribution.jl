abstract type AbstractReparametrizableDistribution <: ContinuousMultivariateDistribution end

# fixed(source::AbstractReparametrizableDistribution) = source.fixed
# variable(source::AbstractReparametrizableDistribution) = source.variable
LogDensityProblems.dimension(source::AbstractReparametrizableDistribution) = length(source)
LogDensityProblems.logdensity(source::AbstractReparametrizableDistribution, draw::AbstractVector) = WarmupHMC.logdensity_and_stuff(source, draw).lpdf

default_info(::AbstractReparametrizableDistribution) = NamedTuple()
info(source::AbstractReparametrizableDistribution) = merge(default_info(source), source.info)
# Base.length(source::AbstractReparametrizableDistribution) = sum(length.(info(source)))
# views(source::AbstractReparametrizableDistribution, draw::AbstractVector) = views(info(source), draw)
# reparametrization_parameters(source::AbstractReparametrizableDistribution) = map(reparametrization_parameters, info(source))
# reparametrize(source::R2D2, parameters::AbstractVector) = R2D2(map(
#     reparametrize, info(source), views(reparametrization_parameters(source), parameters)
# ))