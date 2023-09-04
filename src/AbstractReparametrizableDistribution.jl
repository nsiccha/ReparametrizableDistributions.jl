abstract type AbstractReparametrizableDistribution <: ContinuousMultivariateDistribution end

fixed(source::AbstractReparametrizableDistribution) = source.fixed
variable(source::AbstractReparametrizableDistribution) = source.variable
LogDensityProblems.dimension(source::AbstractReparametrizableDistribution) = length(source)
LogDensityProblems.logdensity(source::AbstractReparametrizableDistribution, draw::AbstractVector) = WarmupHMC.logdensity_and_stuff(source, draw)[1]