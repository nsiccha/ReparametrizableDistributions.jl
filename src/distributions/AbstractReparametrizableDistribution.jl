abstract type AbstractReparametrizableDistribution <: ContinuousMultivariateDistribution end
Broadcast.broadcastable(source::AbstractReparametrizableDistribution) = Ref(source)

Base.getproperty(source::T, key::Symbol) where {T<:AbstractReparametrizableDistribution} = hasfield(T, key) ? getfield(source, key) : getproperty(info(source), key)
info(source::AbstractReparametrizableDistribution) = getfield(source, :info)

Base.length(source::AbstractReparametrizableDistribution) = sum(lengths(source))
lengths(source::AbstractReparametrizableDistribution) = map(length, parts(source))
# IMPLEMENT THIS
parts(::AbstractReparametrizableDistribution) = error("unimplemented")
# IMPLEMENTING THIS FOR WarmupHMC.jl
to_nt(source::AbstractReparametrizableDistribution, draw::AbstractArray) = views(
    parts(source), draw
)
to_array_limited(source, draw) = view(to_array(source, draw), 1:length(source))
to_array(source::AbstractReparametrizableDistribution, draw::NamedTuple) = vcat(
    kmap(to_array_limited, parts(source), draw)...
)
# IMPLEMENT THIS
reparametrization_parameters(::AbstractReparametrizableDistribution) = error("unimplemented")
# IMPLEMENTING THIS FOR WarmupHMC.jl
optimization_reparametrization_parameters(source::AbstractReparametrizableDistribution) = vcat(
    map(
        broadcast, 
        ensure_like(reparametrization_parameters(source), optimization_parameters_fn(source)), 
        reparametrization_parameters(source)
    )...
)
# MAY IMPLEMENT THIS
optimization_parameters_fn(::AbstractReparametrizableDistribution) = identity
# IMPLEMENTING THIS FOR WarmupHMC.jl
reparametrize(source::AbstractReparametrizableDistribution, parameters::AbstractVector) = reparametrize(
    source, 
    map(
        broadcast, 
        map(inverse, ensure_like(reparametrization_parameters(source), optimization_parameters_fn(source))), 
        views(reparametrization_parameters(source), parameters)
    )
)
# MAY IMPLEMENT THIS or THE ABOVE
reparametrize(source::T, parameters::NamedTuple) where {T<:AbstractReparametrizableDistribution} = T.name.wrapper(merge(info(source), parameters))
# IMPLEMENT THIS or THE ABOVE
# reparametrize(::AbstractReparametrizableDistribution, ::NamedTuple) = error("unimplemented")
# MAY IMPLEMENT THIS
# to_array(::AbstractReparametrizableDistribution, ::NamedTuple)
# IMPLEMENT THIS
lpdf_update(::AbstractReparametrizableDistribution, ::NamedTuple, lpdf=0.) = error("unimplemented")
# IMPLEMENT THIS
lja_update(::AbstractReparametrizableDistribution, ::AbstractReparametrizableDistribution, ::NamedTuple, lpdf=0.) = error("unimplemented")

# IMPLEMENTING THIS FOR WarmupHMC.jl
find_reparametrization(source::AbstractReparametrizableDistribution, draw::AbstractVector{<:NamedTuple}; kwargs...) = 
    find_reparametrization(:Optim, source, draw; kwargs...)

find_reparametrization(source::AbstractReparametrizableDistribution, draws::AbstractMatrix; kwargs...) = recombine(
    source, kmap(find_reparametrization, divide(source, draws)...; kwargs...)
)
# MAY IMPLEMENT THIS
divide(source, draws::AbstractMatrix) = divide(
    source, lpdf_and_invariants(source, draws, Ignore())
)
# MAY IMPLEMENT THIS
divide(source, draws::AbstractVector{<:NamedTuple}) = (source, ), (draws, )
# MAY IMPLEMENT THIS
recombine(::Any, resources::NTuple{1}) = resources[1]
# recombine(::Any, ::Any) = error("unimplemented")


# IMPLEMENTING THIS FOR LogDensityProblems.jl
LogDensityProblems.dimension(source::AbstractReparametrizableDistribution) = length(source)
LogDensityProblems.logdensity(source::AbstractReparametrizableDistribution, draw::AbstractVector) = try 
    lpdf_and_invariants(source, draw).lpdf
catch e
    @warn """
Failed to evaluate log density: 
$source
$draw
$(WarmupHMC.exception_to_string(e))
    """
    -Inf
end