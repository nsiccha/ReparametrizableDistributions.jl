using SpecialFunctions, HypergeometricFunctions, ChainRulesCore

struct GammaSimplex{I} <: AbstractReparametrizableDistribution
    info::I
end
GammaSimplex(concentration::AbstractVector) = GammaSimplex(Dirichlet(concentration))
GammaSimplex(target::Dirichlet) = GammaSimplex(target, target)
GammaSimplex(target::Dirichlet, parametrization::Dirichlet) = GammaSimplex((
    target_distribution=target,
    parametrization_distribution=parametrization,
    parametrization_gammas=Gamma.(parametrization.alpha),
    sum_gamma=Gamma(sum(parametrization.alpha)),
))
Base.length(source::GammaSimplex) = length(info(source).target_distribution)
reparametrization_parameters(source::GammaSimplex) = log.(info(source).parametrization_distribution.alpha)
reparametrize(source::GammaSimplex, parameters::AbstractVector) = GammaSimplex(
    info(source).target_distribution, Dirichlet(exp.(parameters))
)

_cdf(distribution, x) = cdf(distribution, x)
_quantile(distribution, x) = quantile(distribution, x)
_logcdf(distribution, x) = logcdf(distribution, x)
_invlogcdf(distribution, x) = invlogcdf(distribution, x)

lpdf_and_invariants(source::GammaSimplex, draw::NamedTuple, lpdf=0.) = begin 
    _info = info(source)
    # _views = views(source, draw)
    unnormalized_weights = _invlogcdf.(_info.parametrization_gammas, _logcdf.(Normal(), draw)) 
    weights = unnormalized_weights ./ sum(unnormalized_weights)
    lpdf += sum(logpdf.(Normal(), draw)) 
    lpdf += logpdf(_info.target_distribution, weights) 
    lpdf -= logpdf(_info.parametrization_distribution, weights)
    (;lpdf, unnormalized_weights, weights)
end

lja_reparametrize(source::GammaSimplex, target::GammaSimplex, draw::NamedTuple, lja=0.) = begin 
    _info = info(source)
    tinfo = info(target)
    sxi = draw.unnormalized_weights#_invlogcdf.(_info.parametrization_gammas, _logcdf.(Normal(), draw))
    ssum = sum(sxi)
    tsum = _invlogcdf(tinfo.sum_gamma, _logcdf(_info.sum_gamma, ssum))
    txi = sxi .* tsum ./ ssum
    tdraw = _invlogcdf.(Normal(), _logcdf.(tinfo.parametrization_gammas, txi))
    lja += sum(logpdf.(Normal(), tdraw))
    lja -= logpdf(tinfo.parametrization_distribution, txi ./ tsum)
    lja, tdraw
end

function ChainRulesCore.rrule(::typeof(_logcdf), d, x::Real)
    lq = _logcdf(d, x)
    function _logcdf_pullback(a)
        q = exp(lq)
        la = a / q
        pullback(grad) = la * grad
        da = @thunk(Tangent{typeof(d)}(;map(pullback, grad_d_cdf(d, x, q))...))
        # da = @thunk(la * Tangent{typeof(d)}(;grad_d_cdf(d, x, q)...))
        xa = pullback(pdf(d, x))
        ChainRulesCore.NoTangent(), da, xa
    end
    lq, _logcdf_pullback
end
grad_d_cdf(d::Gamma, x, q) = begin 
    a, b = params(d)
    xi = x / b
    # https://discourse.julialang.org/t/gamma-inc-derivatives/93148
    g = gamma(a)
    dg = digamma(a)
    lx = log(xi)
    r = pFq([a,a],[a+1,a+1],-xi)
    grad_a = a^(-2) * xi^a * r/g + q*(dg - lx)
    # # https://www.wolframalpha.com/input?i=D%5BGammaRegularized%5Ba%2C+0%2C+x%2Fb%5D%2C+b%5D
    grad_b = -exp(-xi) * xi^a / (b * gamma(a))
    (α=-grad_a::Float64, θ=grad_b::Float64)
end

function ChainRulesCore.rrule(::typeof(_invlogcdf), d, lq::Real)
    x = _invlogcdf(d, lq)
    function _invlogcdf_pullback(a)
        q = exp(lq)
        difdx = q/pdf(d,x)
        pullback(grad) = a*grad/pdf(d,x)
        da = @thunk(-Tangent{typeof(d)}(;map(pullback, grad_d_cdf(d, x, q))...))
        lqa = a * difdx
        ChainRulesCore.NoTangent(), da, lqa 
    end
    x, _invlogcdf_pullback
end