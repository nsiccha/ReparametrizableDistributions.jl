module ReparametrizableDistributions

export ScaleHierarchy, GammaSimplex, HSGP, R2D2

using WarmupHMC, Distributions, LogDensityProblems, LogExpFunctions

import WarmupHMC: reparametrization_parameters, unconstrained_reparametrization_parameters, reparametrize, unconstrained_reparametrize, lpdf_and_invariants, lja_reparametrize

include("StackedVector.jl")
include("AbstractReparametrizableDistribution.jl")
include("ScaleHierarchy.jl")
include("GammaSimplex.jl")
include("HSGP.jl")
include("R2D2.jl")

end # module ReparametrizableDistributions
