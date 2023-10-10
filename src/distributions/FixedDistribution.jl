struct FixedDistribution{W} <: AbstractReparametrizableDistribution
    wrapped::W
end

Base.length(source::FixedDistribution) = length(source.wrapped)
reparametrize(source::FixedDistribution, ::Any) = source
lja_reparametrize(::FixedDistribution, ::FixedDistribution, draw::AbstractVector, lja=0.) = 
lja, draw
lpdf_and_invariants(source::FixedDistribution, draw::AbstractVector, lpdf=0.) = lpdf_and_invariants(
    source.wrapped, draw, lpdf
)
find_reparametrization(source::FixedDistribution, ::Any; kwargs...) = source