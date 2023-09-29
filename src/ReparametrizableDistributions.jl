module ReparametrizableDistributions

export LocScaleHierarchy, ScaleHierarchy, MeanShift, GammaSimplex, HSGP, R2D2, Directional, ReparametrizablePosterior

using WarmupHMC, Distributions, LogDensityProblems, LogExpFunctions
using SpecialFunctions, HypergeometricFunctions, ChainRulesCore

import WarmupHMC: reparametrization_parameters, reparametrize, lpdf_and_invariants, lja_reparametrize, find_reparametrization

include("utils/StackedVector.jl")
include("utils/finite_unconstraining.jl")
include("utils/quantile_and_cdf.jl")
include("distributions/AbstractReparametrizableDistribution.jl")
include("distributions/ScaleHierarchy.jl")
include("distributions/MeanShift.jl")
include("distributions/Directional.jl")
include("distributions/GammaSimplex.jl")
include("distributions/HSGP.jl")
include("distributions/R2D2.jl")
include("distributions/ReparametrizablePosterior.jl")

end # module ReparametrizableDistributions
