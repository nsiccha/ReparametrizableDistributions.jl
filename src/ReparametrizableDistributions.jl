module ReparametrizableDistributions

export ScaleHierarchy, MeanShift, GammaSimplex, HSGP, R2D2

using WarmupHMC, Distributions, LogDensityProblems, LogExpFunctions

import WarmupHMC: reparametrization_parameters, reparametrize, lpdf_and_invariants, lja_reparametrize

include("StackedVector.jl")
include("AbstractReparametrizableDistribution.jl")
include("ScaleHierarchy.jl")
include("MeanShift.jl")
include("GammaSimplex.jl")
include("HSGP.jl")
include("R2D2.jl")

end # module ReparametrizableDistributions
