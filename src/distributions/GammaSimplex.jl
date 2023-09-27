struct GammaSimplex{I} <: AbstractReparametrizableDistribution
    info::I
end
GammaSimplex(target::AbstractVector) = GammaSimplex(Dirichlet(target))
GammaSimplex(target::AbstractVector, parametrization::AbstractVector) = GammaSimplex(
    Dirichlet(target), Dirichlet(parametrization)
)
GammaSimplex(target::Dirichlet) = GammaSimplex(target, target)
GammaSimplex(target::Dirichlet, parametrization::Dirichlet) = GammaSimplex((
    target_distribution=target,
    parametrization_distribution=parametrization,
    parametrization_gammas=Gamma.(parametrization.alpha),
    sum_gamma=Gamma(sum(parametrization.alpha)),
))
length_info(source::GammaSimplex) = Length((xi=length(source.info.target_distribution),))
reparametrization_parameters(source::GammaSimplex) = log.(source.info.parametrization_distribution.alpha)
reparametrize(source::GammaSimplex, parameters::AbstractVector) = GammaSimplex(
    source.info.target_distribution, Dirichlet(exp.(parameters))
)

lpdf_and_invariants(source::GammaSimplex, draw::NamedTuple, lpdf=0.) = begin 
    _info = info(source)
    # _views = views(source, draw)
    unnormalized_weights = quantile_cdf.(_info.parametrization_gammas, Normal(), draw.xi) 
    weights_sum = sum(unnormalized_weights)
    weights = unnormalized_weights ./ weights_sum
    lpdf += sum_logpdf(Normal(), draw.xi) 
    lpdf += logpdf(_info.target_distribution, weights) 
    lpdf -= logpdf(_info.parametrization_distribution, weights)
    (;lpdf, weights, weights_sum)
end

lja_reparametrize(source::GammaSimplex, target::GammaSimplex, invariants::NamedTuple, lja=0.) = begin 
    _info = info(source)
    tinfo = info(target)
    weights = invariants.weights#_invlogcdf.(_info.parametrization_gammas, _logcdf.(Normal(), draw))
    tweights_sum = quantile_cdf(tinfo.sum_gamma, _info.sum_gamma, invariants.weights_sum) 
    tunnormalized_weights = weights .* tweights_sum
    # ssum = sum(sxi)
    # tsum = _invlogcdf(tinfo.sum_gamma, _logcdf(_info.sum_gamma, ssum))
    # txi = sxi .* tsum ./ ssum
    tdraw = quantile_cdf.(Normal(), tinfo.parametrization_gammas, tunnormalized_weights)
    lja += sum_logpdf(Normal(), tdraw)
    lja -= logpdf(tinfo.parametrization_distribution, weights)
    lja, tdraw
end

