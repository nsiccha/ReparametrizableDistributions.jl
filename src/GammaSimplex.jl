using SpecialFunctions, HypergeometricFunctions, ChainRulesCore

struct GammaSimplex{F,V} <: AbstractReparametrizableDistribution
    fixed::F
    variable::V
end
GammaSimplex(dirichlet::Dirichlet) = GammaSimplex(dirichlet, dirichlet)
WarmupHMC.reparametrization_parameters(source::GammaSimplex) = parametrization_concentrations(source)
WarmupHMC.unconstrained_reparametrization_parameters(source::GammaSimplex) = log.(parametrization_concentrations(source))
WarmupHMC.reparametrize(source::GammaSimplex, parameters::AbstractVector) = GammaSimplex(
    fixed(source), Dirichlet(parameters)
)
WarmupHMC.unconstrained_reparametrize(source::GammaSimplex, parameters::AbstractVector) = GammaSimplex(
    fixed(source), Dirichlet(exp.(parameters))
)

target_distribution(source::GammaSimplex) = fixed(source)
Base.length(source::GammaSimplex) = length(target_distribution(source))
parametrization_distribution(source::GammaSimplex) = variable(source)
parametrization_concentrations(source::GammaSimplex) = parametrization_distribution(source).alpha
parametrization_gammas(source::GammaSimplex) = Gamma.(parametrization_concentrations(source))
sum_gamma(source::GammaSimplex) = Gamma(sum(parametrization_concentrations(source)))

_cdf(distribution, x) = cdf(distribution, x)
_quantile(distribution, x) = quantile(distribution, x)
_logcdf(distribution, x) = logcdf(distribution, x)
_invlogcdf(distribution, x) = invlogcdf(distribution, x)


WarmupHMC.logdensity_and_stuff(source::GammaSimplex, draw::AbstractVector, lpdf=0.) = begin 
    lpdf += sum(logpdf.(Normal(), draw)) 
    xi = _invlogcdf.(parametrization_gammas(source), _logcdf.(Normal(), draw))
    x = xi ./ sum(xi)
    lpdf += logpdf(target_distribution(source), x) 
    lpdf -= logpdf(parametrization_distribution(source), x)
    lpdf, x
end

WarmupHMC.lja_reparametrize(source::GammaSimplex, target::GammaSimplex, draw::AbstractVector, lja=0.) = begin 
    sxi = _invlogcdf.(parametrization_gammas(source), _logcdf.(Normal(), draw))
    ssum = sum(sxi)
    tsum = _invlogcdf(sum_gamma(target), _logcdf(sum_gamma(source), ssum))
    txi = sxi .* tsum ./ ssum
    tdraw = _invlogcdf.(Normal(), _logcdf.(parametrization_gammas(target), txi))
    lja += sum(logpdf.(Normal(), tdraw))
    lja -= logpdf(parametrization_distribution(target), txi ./ tsum)
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