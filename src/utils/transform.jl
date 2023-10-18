struct LogTransformed{D} <: ContinuousUnivariateDistribution
    distribution::D
end

log_transform(d) = LogTransformed(d)
Distributions.logpdf(source::LogTransformed, x::Real) = x + logpdf(source.distribution, exp(x))