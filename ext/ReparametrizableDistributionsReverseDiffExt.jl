module ReparametrizableDistributionsReverseDiffExt

using ReparametrizableDistributions, ReverseDiff, Distributions

import ReparametrizableDistributions: _logcdf, _invlogcdf

import ReverseDiff: TrackedReal
ReverseDiff.@grad_from_chainrules _logcdf(d, x::TrackedReal)
ReverseDiff.@grad_from_chainrules _invlogcdf(d, x::TrackedReal)
ReverseDiff.value(d::Gamma{<:TrackedReal}) = Gamma(ReverseDiff.value.(params(d))...)
# ReverseDiff.@grad_from_chainrules logcdf(d::Gamma{<:TrackedReal}, x::Real)
ReverseDiff.@grad_from_chainrules _invlogcdf(d::Gamma{<:TrackedReal}, x::Real)

ReverseDiff.value(d::NoncentralChisq{<:TrackedReal}) = NoncentralChisq(ReverseDiff.value.(params(d))...)
ReverseDiff.@grad_from_chainrules _invlogcdf(d::NoncentralChisq{<:TrackedReal}, x::Real)

end