struct Directional{I} <: AbstractReparametrizableDistribution
    info::I
end

finite_log(x, reg=1e-16) = log(x + reg)

Directional(dimension, non_centrality) = Directional(
    (;
        dimension, 
        non_centrality,
        radius_squared=NoncentralChisq(dimension, non_centrality), 
        # radius_squared=truncated(Normal(non_centrality, 1); lower=0)
    )
)
length_info(source::Directional) = Length((location=source.info.dimension,))
reparametrization_parameters(source::Directional) = [finite_log(source.info.non_centrality)]
reparametrize(source::Directional, parameters::AbstractVector) = Directional(source.info.dimension, exp(parameters[1]))

lpdf_and_invariants(source::Directional, draw::NamedTuple, lpdf=0.) = begin
    _info = info(source)
    radius_squared = 1e-8 + sum(draw.location .^ 2)
    direction = draw.location ./ sqrt(radius_squared)
    lpdf += sum_logpdf(Normal(), draw.location)
    lpdf += (
        _logpdf(_info.radius_squared, radius_squared) 
        - _logpdf(Chisq(_info.dimension), radius_squared)
    )
    (;lpdf, radius_squared, direction)
end

lja_reparametrize(source::Directional, target::Directional, invariants::NamedTuple, lja=0.) = begin 
    _info = info(source)
    tinfo = info(target)

    tradius_squared = quantile_cdf(
        tinfo.radius_squared, _info.radius_squared, invariants.radius_squared
    )
    tlocation = invariants.direction .* sqrt(tradius_squared)
    lja += sum_logpdf(Normal(), tlocation)
    lja += (
        _logpdf(tinfo.radius_squared, tradius_squared) 
        - _logpdf(Chisq(tinfo.dimension), tradius_squared)
    )

    lja, tlocation
end