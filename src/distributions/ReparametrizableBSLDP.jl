WarmupHMC.reparametrize(::ADGradientWrapper, target::AbstractReparametrizableDistribution) = begin
    ADgradient(:ReverseDiff, target)
end

struct ReparametrizableBSLDP{F,P} <: ADGradientWrapper
    stan_file::AbstractString
    posterior::StanModel
    model_function::F
    _posterior::P
end
Broadcast.broadcastable(source::ReparametrizableBSLDP) = Ref(source)
ReparametrizableBSLDP(stan_file, model_function, data) = ReparametrizableBSLDP(
    stan_file,
    StanModel(;stan_file=stan_file, data=data),
    model_function,
    model_function(JSON.parsefile(data))
)

LogDensityProblems.dimension(what::ReparametrizableBSLDP) = Int64(BridgeStan.param_unc_num(what.posterior))
LogDensityProblems.capabilities(::Type{<:ReparametrizableBSLDP}) = LogDensityProblems.LogDensityOrder{2}()
LogDensityProblems.logdensity(what::ReparametrizableBSLDP, x) = try 
    BridgeStan.log_density(what.posterior, collect(x))
catch e
    @warn """
Failed to evaluate log density: 
$what
$x
$(WarmupHMC.exception_to_string(e))
    """
    -Inf
end
LogDensityProblems.logdensity_and_gradient(source::ReparametrizableBSLDP, draw) = try 
    BridgeStan.log_density_gradient(source.posterior, collect(draw))
catch e
    @warn """
Failed to evaluate log density gradient: 
$source
$draw
$(WarmupHMC.exception_to_string(e))
    """
    -Inf, -Inf .* draw
end
LogDensityProblems.logdensity_gradient_and_hessian(what::ReparametrizableBSLDP, x) = BridgeStan.log_density_hessian(what.posterior, collect(x))

Base.parent(what::ReparametrizableBSLDP) = what._posterior

update_dict(::Any, ::Any) = error("unimplemented")
WarmupHMC.reparametrize(source::ReparametrizableBSLDP, target::AbstractReparametrizableDistribution) = begin
    data = merge(JSON.parse(source.posterior.data), update_dict(source.model_function, target))
    ReparametrizableBSLDP(
        source.stan_file,
        StanModel(;stan_file=source.stan_file, data=JSON.json(data)),
        source.model_function,
        source.model_function(data)
    )
end