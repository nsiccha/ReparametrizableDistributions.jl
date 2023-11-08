abstract type AbstractDirectional <: AbstractReparametrizableDistribution end
parts(source::AbstractDirectional) = (;source.direction)

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
reparametrization_parameters(source::NormalDirectional) = (;source.c1, source.c2)
optimization_parameters_fn(::NormalDirectional) = finite_log
reparametrize(source::NormalDirectional, parameters::NamedTuple) = NormalDirectional(source.dimension, parameters...)

lpdf_update(source::AbstractDirectional, draw::NamedTuple, lpdf=0.) = begin
    radius_squared = 1e-8 + sum(draw.direction .^ 2)
    direction = draw.direction ./ sqrt(radius_squared)
    lpdf += sum_logpdf(Normal(), draw.direction)
    lpdf += (
        _logpdf(source.radius_squared, radius_squared) 
        - _logpdf(Chisq(source.dimension), radius_squared)
    )
    (;lpdf, direction, radius_squared)
end

lja_update(source::AbstractDirectional, target::AbstractDirectional, invariants::NamedTuple, lja=0.) = begin 
    radius_squared = quantile_cdf(
        target.radius_squared, source.radius_squared, invariants.radius_squared
    )
    direction = invariants.direction .* sqrt(radius_squared)
    lja += sum_logpdf(Normal(), direction)
    lja += (
        _logpdf(target.radius_squared, radius_squared) 
        - _logpdf(Chisq(target.dimension), radius_squared)
    )

    (;lja, direction, radius_squared)
end

# struct Directional{I} <: AbstractDirectional
#     info::I
# end

# Directional(dimension, non_centrality) = Directional(
#     (;
#         dimension, 
#         non_centrality,
#         radius_squared=NoncentralChisq(dimension, non_centrality), 
#     )
# )
# reparametrize(source::Directional, parameters::AbstractVector) = Directional(source.info.dimension, exp(parameters[1]))