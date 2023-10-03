struct ReparametrizableBSLDP{F,P} <: ADGradientWrapper
    stan_file::AbstractString
    posterior::StanModel
    model_function::F
    _posterior::P
end
ReparametrizableBSLDP(stan_file, model_function, data) = ReparametrizableBSLDP(
    stan_file,
    StanModel(;stan_file=stan_file, data=data),
    model_function,
    model_function(JSON.parsefile(data))
)

LogDensityProblems.dimension(what::ReparametrizableBSLDP) = Int64(BridgeStan.param_unc_num(what.posterior))
LogDensityProblems.capabilities(::Type{<:ReparametrizableBSLDP}) = LogDensityProblems.LogDensityOrder{2}()
LogDensityProblems.logdensity(what::ReparametrizableBSLDP, x) = try 
    BridgeStan.log_density(what.posterior, x)
catch e
    -Inf
end
LogDensityProblems.logdensity_and_gradient(what::ReparametrizableBSLDP, x) = try 
    BridgeStan.log_density_gradient(what.posterior, x)
catch e
    -Inf, -Inf .* x
end
LogDensityProblems.logdensity_gradient_and_hessian(what::ReparametrizableBSLDP, x) = BridgeStan.log_density_hessian(what.posterior, x)

Base.parent(what::ReparametrizableBSLDP) = what._posterior
WarmupHMC.reparametrize(source::ReparametrizableBSLDP, target::AbstractReparametrizableDistribution) = begin
    data = merge(JSON.parse(source.posterior.data), update_dict(source.model_function, target))
    ReparametrizableBSLDP(
        source.stan_file,
        StanModel(;stan_file=source.stan_file, data=JSON.json(data)),
        source.model_function,
        source.model_function(data)
    )
end