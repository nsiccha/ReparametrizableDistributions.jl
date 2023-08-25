module ReparametrizableDistributionsReverseDiffExt

using ReparametrizableDistributions, ReverseDiff

import ReparametrizableDistributions: _cdf, _quantile, _logcdf, _invlogcdf

_cdf(distribution, x::ReverseDiff.TrackedReal) = ReverseDiff.track(_cdf, distribution, x)
ReverseDiff.@grad function _cdf(distribution, tx)
    x = ReverseDiff.value.(tx)
    q = _cdf.(distribution, x)
    return q, a -> (nothing, a .* pdf.(distribution, x))
end
_quantile(distribution, q::ReverseDiff.TrackedReal) = ReverseDiff.track(_quantile, distribution, q)
ReverseDiff.@grad function _quantile(distribution, tq)
    x = _quantile.(distribution, ReverseDiff.value(tq))
    return x, a -> (nothing, a ./ pdf.(distribution, x))
end
_logcdf(distribution, x::ReverseDiff.TrackedReal) = ReverseDiff.track(_logcdf, distribution, x)
ReverseDiff.@grad function _logcdf(distribution, tx)
    x = ReverseDiff.value.(tx)
    lq = _logcdf.(distribution, x)
    return lq, a -> (nothing, a .* exp.(logpdf.(distribution, x) - lq))
end
_invlogcdf(distribution, x::ReverseDiff.TrackedReal) = ReverseDiff.track(_invlogcdf, distribution, x)
ReverseDiff.@grad function _invlogcdf(distribution, tlq)
    lq = ReverseDiff.value.(tlq)
    x = _invlogcdf.(distribution, lq)
    return x, a -> (nothing, a .* exp.(lq - logpdf.(distribution, x)))
end

end