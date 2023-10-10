abstract type AbstractDirectional <: AbstractReparametrizableDistribution end


lengths(source::AbstractDirectional) = (location=source.dimension,)
# reparametrization_parameters(source::AbstractDirectional) = [finite_log(source.info.non_centrality)]

lpdf_and_invariants(source::AbstractDirectional, draw::NamedTuple, lpdf=0.) = begin
    radius_squared = 1e-8 + sum(draw.location .^ 2)
    direction = draw.location ./ sqrt(radius_squared)
    lpdf += sum_logpdf(Normal(), draw.location)
    lpdf += (
        _logpdf(source.radius_squared, radius_squared) 
        - _logpdf(Chisq(source.dimension), radius_squared)
    )
    (;lpdf, radius_squared, direction)
end

lja_reparametrize(source::AbstractDirectional, target::AbstractDirectional, invariants::NamedTuple, lja=0.) = begin 
    tradius_squared = quantile_cdf(
        target.radius_squared, source.radius_squared, invariants.radius_squared
    )
    tlocation = invariants.direction .* sqrt(tradius_squared)
    lja += sum_logpdf(Normal(), tlocation)
    lja += (
        _logpdf(target.radius_squared, tradius_squared) 
        - _logpdf(Chisq(target.dimension), tradius_squared)
    )

    (;lja, draw=tlocation)
end

struct Directional{I} <: AbstractDirectional
    info::I
end

Directional(dimension, non_centrality) = Directional(
    (;
        dimension, 
        non_centrality,
        radius_squared=NoncentralChisq(dimension, non_centrality), 
    )
)
reparametrize(source::Directional, parameters::AbstractVector) = Directional(source.info.dimension, exp(parameters[1]))

struct NormalDirectional{I} <: AbstractDirectional
    info::I
end

NormalDirectional(dimension, c1, c2) = NormalDirectional(
    (;
        dimension, 
        c1, c2,
        radius_squared=truncated(Normal(c1, c2); lower=0)
    )
)
reparametrization_parameters(source::NormalDirectional) = finite_log.([source.info.c1, source.info.c2])
reparametrize(source::NormalDirectional, parameters::AbstractVector) = NormalDirectional(source.info.dimension, exp.(parameters)...)