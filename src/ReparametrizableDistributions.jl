module ReparametrizableDistributions

export ScaleHierarchy, MeanShift, GammaSimplex, HSGP, R2D2, Directional

using WarmupHMC, Distributions, LogDensityProblems, LogExpFunctions
using SpecialFunctions, HypergeometricFunctions, ChainRulesCore

import WarmupHMC: reparametrization_parameters, reparametrize, lpdf_and_invariants, lja_reparametrize

include("utils/StackedVector.jl")
include("utils/quantile_and_cdf.jl")
include("distributions/AbstractReparametrizableDistribution.jl")
include("distributions/ScaleHierarchy.jl")
include("distributions/MeanShift.jl")
include("distributions/Directional.jl")
include("distributions/GammaSimplex.jl")
include("distributions/HSGP.jl")
include("distributions/R2D2.jl")

end # module ReparametrizableDistributions
