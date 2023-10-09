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

# default_info(::AbstractReparametrizableDistribution) = NamedTuple()
info(source::AbstractReparametrizableDistribution) = source.info#merge(default_info(source), source.info)
length_info(source::AbstractReparametrizableDistribution) = info(source)
reparametrization_info(source::AbstractReparametrizableDistribution) = info(source)
# info_and_views(source::AbstractReparametrizableDistribution, draw::AbstractVector) = info(source), views(source, draw)
Base.length(source::AbstractReparametrizableDistribution) = total_length(length_info(source))
views(source::AbstractReparametrizableDistribution, draw::AbstractArray) = views(length_info(source), draw)
_reparametrization_parameters(source::AbstractReparametrizableDistribution) = StackedArray(
    map(reparametrization_parameters, reparametrization_info(source))
)
reparametrization_parameters(source::AbstractReparametrizableDistribution) = collect(_reparametrization_parameters(source))
reparametrize(source::AbstractReparametrizableDistribution, parameters::AbstractVector) = reparametrize(
    source, views(_reparametrization_parameters(source), parameters)
)
lpdf_and_invariants(source::AbstractReparametrizableDistribution, draw::AbstractVector, lpdf=0.) = lpdf_and_invariants(source, views(source, draw), lpdf)
# lja_reparametrize(source::AbstractReparametrizableDistribution, target::AbstractReparametrizableDistribution, draw::AbstractVector, lja=0.) = try 
#     lja, tdraw = lja_reparametrize(source, target, lpdf_and_invariants(source, draw), lja)
#     # !all(isfinite.(tdraw)) && println((;source, target, draw, lpdf_and_invariants=lpdf_and_invariants(source, draw), tdraw))
#     lja, collect(tdraw)
# catch e
#     @debug exception_to_string(e)
#     NaN, NaN .* draw
# end
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


divide(source::Any, draws::AbstractMatrix) = divide(source, lpdf_and_invariants(source, draws, Ignore()))
divide(source, draws) = (source, ), (draws, )
recombine(::Any, resources::NTuple{1}) = resources[1]
WarmupHMC.find_reparametrization(source::AbstractReparametrizableDistribution, draws; kwargs...) = begin
    recombine(source, WarmupHMC.find_reparametrization.(:Optim, divide(source, draws)...; kwargs...))
end
kmap(f, args...; kwargs...) = map((args...)->f(args...; kwargs...), args...)