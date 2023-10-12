module ReparametrizableDistributions

export LocScaleHierarchy, ScaleHierarchy, MeanShift, GammaSimplex, HSGP, PHSGP, R2D2, Directional, ReparametrizablePosterior, ReparametrizableBSLDP, FixedDistribution

using WarmupHMC, Distributions, LogDensityProblems, LogExpFunctions
using SpecialFunctions, HypergeometricFunctions, ChainRulesCore
using BridgeStan, JSON

import WarmupHMC: reparametrization_parameters, optimization_reparametrization_parameters, reparametrize, lpdf_and_invariants, lja_and_reparametrize, to_array, to_nt, lpdf_update, lja_update, find_reparametrization

import LogDensityProblemsAD: ADgradient, ADGradientWrapper

kmap_(f, args...; kwargs...) = map((args...)->f(args...; kwargs...), args...)
kmap(f, args...; kwargs...) = kmap_(f, args...; kwargs...)
kmap(f, arg::NamedTuple, args...; kwargs...) = kmap_(f, arg, ensure_like.(Ref(arg), args)...; kwargs...)
ensure_like(::NamedTuple{names}, rhs::NamedTuple) where {names} = NamedTuple{names}(rhs)
ensure_like(::NamedTuple{names}, rhs) where {names} = NamedTuple{names}((rhs for name in names))

include("utils/StackedArray.jl")
include("utils/finite_unconstraining.jl")
include("utils/quantile_and_cdf.jl")
include("distributions/AbstractReparametrizableDistribution.jl")
include("distributions/ScaleHierarchy.jl")
include("distributions/MeanShift.jl")
include("distributions/Directional.jl")
include("distributions/GammaSimplex.jl")
include("distributions/AbstractWrappedDistribution.jl")
include("distributions/FixedDistribution.jl")
include("distributions/AbstractCompositeReparametrizableDistribution.jl")
include("distributions/HSGP.jl")
include("distributions/R2D2.jl")
include("distributions/ReparametrizablePosterior.jl")
include("distributions/ReparametrizableBSLDP.jl")

end # module ReparametrizableDistributions
