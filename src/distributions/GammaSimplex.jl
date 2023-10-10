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

lengths(source::GammaSimplex) = (xi=length(source.target_distribution),)
reparametrization_parameters(source::GammaSimplex) = log.(source.parametrization_distribution.alpha)
reparametrize(source::GammaSimplex, parameters::AbstractVector) = GammaSimplex(
    source.target_distribution, Dirichlet(exp.(parameters))
)

lpdf_and_invariants(source::GammaSimplex, draw::NamedTuple, lpdf=0.) = begin 
    unnormalized_weights = quantile_cdf.(source.parametrization_gammas, Normal(), draw.xi) 
    weights_sum = sum(unnormalized_weights)
    weights = unnormalized_weights ./ weights_sum
    lpdf += sum_logpdf(Normal(), draw.xi) 
    lpdf += logpdf(source.target_distribution, weights) 
    lpdf -= logpdf(source.parametrization_distribution, weights)
    (;lpdf, weights, weights_sum)
end
lja_reparametrize(source::GammaSimplex, target::GammaSimplex, invariants::NamedTuple, lja=0.) = begin 
    weights_sum = quantile_cdf(target.sum_gamma, source.sum_gamma, invariants.weights_sum) 
    unnormalized_weights = invariants.weights .* weights_sum
    xi = quantile_cdf.(Normal(), target.parametrization_gammas, unnormalized_weights)
    lja += sum_logpdf(Normal(), xi)
    lja -= logpdf(tinfo.parametrization_distribution, invariants.weights)
    (;lja, xi)
end

