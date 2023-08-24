module ReparametrizableDistributions

using WarmupHMC, Distributions, UnPack

abstract type AbstractReparametrizableDistribution end

WarmupHMC.reparametrization_parameters(source::AbstractReparametrizableDistribution) = vcat(
    
)
AbstractReparametrizableDistribution

struct ReparametrizableDistribution{F,R,D}
    parameters::P
    distribution::D
end

LocScaleHierarchy2 = ReparametrizableDistribution{:LocScaleHierarchy}

LocScaleHierarchy(loc, log_scale, xi, mean_centeredness, sd_centeredness) = LocScaleHierarchy2((;mean_centeredness, sd_centeredness), (;loc, log_scale, xi))

LocScaleHierarchy(1, 2, 3, 4, 5)
# struct LocScaleHierarchy{L,S,P,T}
#     loc::L
#     log_scale::S
#     parameters::P
#     mean_centeredness::Vector{T}
#     sd_centeredness::Vector{T}
# end

# WarmupHMC.reparametrization_parameters(source::LocScaleHierarchy) = vcat(
#     source.mean_centeredness, source.sd_centeredness
# )
# WarmupHMC.reparametrize(source::LocScaleHierarchy, parameters::AbstractVector) = LocScaleHierarchy(
#     source.loc,
#     source.log_scale,
#     source.parameters,
#     collect.(eachcol(reshape(parameters, (:, 2))))...
# )

# struct GammaSimplex{D,T}
#     dirichlet::D
#     concentration::Vector{T}
# end

# struct R2D2{I,S,R,D,T}
#     intercept::I
#     sigma_sq::S
#     R2::R
#     dirichlet::D
#     X::Matrix{T}
# end

end # module ReparametrizableDistributions
