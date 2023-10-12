struct ReparametrizableBSLDP{F,W} <: AbstractWrappedDistribution
    lib_path::AbstractString
    proxy::StanModel
    model_function::F
    wrapped::W
end
ReparametrizableBSLDP(lib_path, model_function, data) = ReparametrizableBSLDP(
    lib_path,
    StanModel(lib_path, data),
    model_function,
    model_function(data)
)

# LogDensityProblems.dimension(source::ReparametrizableBSLDP) = Int64(BridgeStan.param_unc_num(source.proxy))
LogDensityProblems.capabilities(::Type{<:ReparametrizableBSLDP}) = LogDensityProblems.LogDensityOrder{2}()
LogDensityProblems.logdensity(source::ReparametrizableBSLDP, draw) = try 
    BridgeStan.log_density(source.proxy, collect(draw))
catch e
    @warn """
Failed to evaluate log density: 
$source
$draw
$(WarmupHMC.exception_to_string(e))
    """
    -Inf
end
LogDensityProblems.logdensity_and_gradient(source::ReparametrizableBSLDP, draw) = try 
    BridgeStan.log_density_gradient(source.proxy, collect(draw))
catch e
    @warn """
Failed to evaluate log density gradient: 
$source
$draw
$(WarmupHMC.exception_to_string(e))
    """
    -Inf, -Inf .* draw
end
LogDensityProblems.logdensity_gradient_and_hessian(source::ReparametrizableBSLDP, draw) = try 
    BridgeStan.log_density_hessian(source.proxy, collect(draw))
catch e
    @warn """
Failed to evaluate log density gradient: 
$source
$draw
$(WarmupHMC.exception_to_string(e))
    """
    -Inf, -Inf .* draw, -Inf .* draw .* draw'
end
# IMPLEMENT THIS
update_nt(::Any, ::Any) = error("unimplemented")
update_dict(model_function, reparent) = Dict([
    (String(key), value) for (key, value) in pairs(update_nt(model_function, reparent))
])
recombine(source::ReparametrizableBSLDP, reparent) = ReparametrizableBSLDP(
    source.lib_path,
    source.model_function,
    merge(JSON.parse(source.proxy.data), update_dict(source.model_function, reparent))
)
verify(source::ReparametrizableBSLDP, draws::AbstractMatrix) = begin 
    @assert BridgeStan.param_unc_num(source.proxy) == length(source.wrapped)
    proxy_lpdfs = LogDensityProblems.log_density.(source, eachcol(draws))
    wrapped_lpdfs = [lpdf_and_invariants(source, draw).lpdf for draw in eachcol(draws)]
    @assert nanstd(proxy_lpdfs - wrapped_lpdfs) < 1e-8 """
Failed lpdf check: $(hcat(proxy_lpdfs, wrapped_lpdfs))
"""
end