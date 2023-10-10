abstract type AbstractWrappedDistribution <: AbstractReparametrizableDistribution end
Base.parent(source::AbstractWrappedDistribution) = source.wrapped

parts(source::AbstractWrappedDistribution) = parts(parent(source))
reparametrization_parameters(source::AbstractWrappedDistribution) = reparametrization_parameters(parent(source))
reparametrize(source::T, parameters::NamedTuple) where {T<:AbstractWrappedDistribution} = T(reparametrize(parent(source), parameters))
lpdf_update(source::AbstractWrappedDistribution, draw::NamedTuple, lpdf=0.) = lpdf_update(
    parent(source), draw, lpdf
)
lja_update(source::AbstractWrappedDistribution, target::AbstractWrappedDistribution, draw::NamedTuple, lpdf=0.) = lja_update(
    parent(source), parent(target), draw, lpdf
)
find_reparametrization(source::AbstractWrappedDistribution, draws::AbstractMatrix; kwargs...) = find_reparametrization(
    parent(source), draws; kwargs...
)