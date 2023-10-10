abstract type AbstractReparametrizableDistribution <: ContinuousMultivariateDistribution end
Broadcast.broadcastable(source::AbstractReparametrizableDistribution) = Ref(source)

Base.getproperty(source::T, key::Symbol) where {T<:AbstractReparametrizableDistribution} = hasfield(T, key) ? getfield(source, key) : getproperty(info(source), key)

LogDensityProblems.dimension(source::AbstractReparametrizableDistribution) = length(source)
LogDensityProblems.logdensity(source::AbstractReparametrizableDistribution, draw::AbstractVector) = try 
    WarmupHMC.lpdf_and_invariants(source, draw).lpdf
catch e
    @warn """
Failed to evaluate log density: 
$source
$draw
$(WarmupHMC.exception_to_string(e))
    """
    -Inf
end

info(source::AbstractReparametrizableDistribution) = source.info
length_info(::Any) =  error("unimplemented")
reparametrization_info(::Any) =  error("unimplemented")
Base.length(source::AbstractReparametrizableDistribution) = total_length(length_info(source))
views(source::AbstractReparametrizableDistribution, draw::AbstractArray) = views(length_info(source), draw)
# reparametrization_parameters_nt(source::AbstractReparametrizableDistribution) = views(
#     map(reparametrization_parameters, reparametrization_info(source))
# )
reparametrization_parameters(source::AbstractReparametrizableDistribution) = collect(reparametrization_parameters_nt(source))
# reparametrize(source::AbstractReparametrizableDistribution, parameters::AbstractVector) = reparametrize(
    # source, views(_reparametrization_parameters(source), parameters)
# )
lpdf_and_invariants(source::AbstractReparametrizableDistribution, draw::AbstractVector, lpdf=0.) = lpdf_and_invariants(source, views(source, draw), lpdf)


divide(source::Any, draws::AbstractMatrix) = divide(source, lpdf_and_invariants(source, draws, Ignore()))
divide(source, draws) = (source, ), (draws, )
recombine(::Any, resources::NTuple{1}) = resources[1]
WarmupHMC.find_reparametrization(source::AbstractReparametrizableDistribution, draws; kwargs...) = begin
    recombine(source, WarmupHMC.find_reparametrization.(:Optim, divide(source, draws)...; kwargs...))
end