_logpdf(distribution, x) = logpdf(distribution, x)
_cdf(distribution, x) = cdf(distribution, x)
_quantile(distribution, x) = quantile(distribution, x)
_logcdf(distribution, x) = logcdf(distribution, x)
_invlogcdf(distribution, x) = invlogcdf(distribution, x)
quantile_cdf(target, source, x) = _invlogcdf(target, _logcdf(source, x))

# https://github.com/stan-dev/math/blob/9b2e93ba58fa00521275b22a190468ab22f744a3/stan/math/prim/fun/log_modified_bessel_first_kind.hpp#L191-L213
logbesseli(k, z) = begin 
    log_half_z = log(.5 * z)
    lgam = loggamma(k + 1.)
    lcons = (2. + k) * log_half_z
    out = logsumexp([k * log_half_z - lgam, lcons - loggamma(k + 2.)])
    lgam += log1p(k)
    m = 2
    lfac = 0
    while true
        old_out = out
        lfac += log(m)
        lgam += log(k+m)
        lcons += 2 * log_half_z
        out = logsumexp([out, lcons - lfac - lgam])
        m += 1
        (out > old_out || out < old_out) || break
    end
    return out
end

_logpdf(distribution::NoncentralChisq, x::Real) = begin
    k, lambda = distribution.ν, distribution.λ
    try (
        log(2) 
        - (x+lambda)/2 
        + (k/4-.5) * log(x/lambda) 
        # + log(besseli(k/2-1, sqrt(lambda*x)))
        + logbesseli(k/2-1, sqrt(lambda*x))
    )
    catch e
        println(e)
        -Inf
    end
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
grad_d_cdf(d::NoncentralChisq, x, q) = begin
    # https://en.wikipedia.org/wiki/Noncentral_chi-squared_distribution
    k, lambda = params(d)
    nu, a, b = k/2, sqrt(lambda), sqrt(x)
    # https://en.wikipedia.org/wiki/Marcum_Q-function#Differentiation
    grad_a = a * (b/a)^nu*exp(-(a^2+b^2)/2)*besseli(nu,a*b)
    grad_k = 0.
    grad_lambda = -grad_a/(2a)
    (ν=grad_k::Float64, λ=grad_lambda::Float64)
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