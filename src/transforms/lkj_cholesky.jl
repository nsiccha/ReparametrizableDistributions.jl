"""
    LKJCholeskyTransform(L::LowerTriangular; eta=1.0)

Fused Cholesky factor reparametrization + LKJ prior.

Given an `n×n` lower-triangular matrix `L` (used as mutable storage), this transform
consumes `n + n*(n-1)/2` unconstrained parameters and:

1. Writes the constrained Cholesky factor into `L`
2. Accumulates the log prior density (LKJ with concentration `eta`) plus all Jacobian
   corrections

**Parametrization (per row `i`):**
- One `log_scale` parameter (standard normal prior) controlling the row's overall scale
  (i.e. the diagonal of the implied covariance matrix)
- `i-1` unconstrained correlation parameters mapped through `tanh(xi / sqrt(n-j))`
  to `(-1, 1)`, with `sqrt(n-j)` providing soft regularization toward zero correlation

**Storage:** `L` is mutated in place. Wrap a `Cholesky` factorization's `.L` field
or allocate with `LowerTriangular(zeros(n, n))`.
"""
struct LKJCholeskyTransform{T<:AbstractMatrix} <: FusedTransform
    L::T
    eta::Float64
    function LKJCholeskyTransform(L::LowerTriangular; eta=1.0)
        new{typeof(L)}(L, eta)
    end
end

"""
    LKJCholeskyTransform(n::Int; eta=1.0)

Create an `LKJCholeskyTransform` with freshly allocated `n×n` storage.
"""
LKJCholeskyTransform(n::Int; eta=1.0) = LKJCholeskyTransform(LowerTriangular(zeros(n, n)); eta)

"""
    nparams(t::LKJCholeskyTransform)

Number of unconstrained parameters consumed: `n` scales + `n*(n-1)/2` correlations.
"""
nparams(t::LKJCholeskyTransform) = begin
    n = size(t.L, 1)
    n + n * (n - 1) ÷ 2
end

function fused_logdensity(t::LKJCholeskyTransform, x; init=(0.0, 0))
    (; L, eta) = t
    lp, pos = init
    n = LinearAlgebra.checksquare(L)
    log_scale, pos = advance!!(x, pos)
    lp += logpdf(Normal(), log_scale)
    L[1, 1] = exp(log_scale)
    for i in 2:n
        log_scale, pos = advance!!(x, pos)
        lp += logpdf(Normal(), log_scale)
        xi, pos = advance!!(x, pos)
        tmp = log_abs_tanh(xi / sqrt(n - 1))
        L[i, 1] = sign(xi) * exp(log_scale + tmp)
        log_sos = 2 * tmp
        lp += log1mexp(log_sos)
        for j in 2:i-1
            xi, pos = advance!!(x, pos)
            tmp1 = 0.5 * log1mexp(log_sos)
            lp += tmp1
            tmp2 = log_abs_tanh(xi / sqrt(n - j))
            lp += log1mexp(2 * tmp2)
            tmp = tmp1 + tmp2
            L[i, j] = sign(xi) * exp(log_scale + tmp)
            log_sos = logaddexp(log_sos, 2 * tmp)
        end
        L[i, i] = exp(log_scale + 0.5 * log1mexp(log_sos))
        lp += (n - i + 2 * eta - 2) * 0.5 * log1mexp(log_sos)
    end
    lp, pos
end
