module ReparametrizableDistributions

export ScaleHierarchy, GammaSimplex, R2D2

using WarmupHMC, Distributions, LogDensityProblems

import WarmupHMC: reparametrization_parameters, unconstrained_reparametrization_parameters, reparametrize, unconstrained_reparametrize, logdensity_and_stuff, lja_reparametrize

include("StackedVector.jl")
include("AbstractReparametrizableDistribution.jl")
include("ScaleHierarchy.jl")
include("GammaSimplex.jl")
include("R2D2.jl")

end # module ReparametrizableDistributions
