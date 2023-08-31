struct R2D2{F,V} <: AbstractReparametrizableDistribution
    fixed::F
    variable::V
end
WarmupHMC.reparametrization_parameters(source::R2D2) = variable(source)
WarmupHMC.reparametrize(source::R2D2, parameters::AbstractVector) = R2D2(
    fixed(source), oftype(source.parameters, parameters)
)

# centeredness(source::R2D2) = variable(source)
Base.length(source::R2D2) = 3+length(variable(source))

# log_scale_distribution(source::R2D2, ::Any) = Product([fixed(source)])
# hierarchical_distribution(source::R2D2, draw::AbstractVector) = begin 
#     log_scale = draw[1]
#     scales = exp.(log_scale .* centeredness(source))
#     Product(Normal.(0, scales))
# end
# subdistributions(source::R2D2, draw::AbstractVector) = log_scale_distribution(source, draw), hierarchical_distribution(source, draw)

@views WarmupHMC.logdensity_and_stuff(source::R2D2, draw::AbstractVector, lpdf=0.) = begin
    intercept, sigma_sq, R2 = draw[1:3]
    simplex_draw, hierarchical_draw = reshape(draw[4:end], (:, 2))
    simplex_lpdf, simplex_x = WarmupHMC.logdensity_and_stuff(simplex(source), simplex_draw)
    lpdf += simplex_lpdf
    sigma = sqrt(sigma_sq)
    tau = R2 / (1 - R2)
    sds = (sigma*tau) .* sqrt.(simplex_x)
    lpdf += sum(logpdf.(Normal(0, sds .^ centeredness(source)))
    beta = hierarchical_draw .* sds .^ (1 - centeredness(source))
    # hierarchical_lpdf, hierarchical_draw = WarmupHMC.logdensity_and_stuff
    # lpdf += hierarchical_lpdf
    lpdf, beta
end

WarmupHMC.lja_reparametrize(source::R2D2, target::R2D2, draw::AbstractVector, lja=0.) = begin 
    lja, tdraw
end