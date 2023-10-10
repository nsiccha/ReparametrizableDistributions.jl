struct FixedDistribution{W} <: AbstractWrappedDistribution
    wrapped::W
end

reparametrize(source::FixedDistribution, ::Any) = source
find_reparametrization(source::FixedDistribution, ::AbstractMatrix; kwargs...) = source