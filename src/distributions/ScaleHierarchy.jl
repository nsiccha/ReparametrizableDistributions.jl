# abstract type AbstractLocScaleHierarchy <: AbstractReparametrizableDistribution end

struct ScaleHierarchy{I} <: AbstractReparametrizableDistribution
    info::I
end
ScaleHierarchy(log_scale, centeredness) = ScaleHierarchy((;log_scale, centeredness))

lengths(source::ScaleHierarchy) = (log_scale=length(source.log_scale), xic=length(source.centeredness))
reparametrization_parameters(source::ScaleHierarchy) = finite_logit.(source.centeredness)
reparametrize(source::ScaleHierarchy, parameters::AbstractVector) = ScaleHierarchy(source.log_scale, logistic.(parameters))

lpdf_and_invariants(source::ScaleHierarchy, draw::NamedTuple, lpdf=0.) = lpdf_and_invariants(
    LocScaleHierarchy([], source.log_scale, source.centeredness), (;location=0., draw...), lpdf
)

divide(source::ScaleHierarchy, draws::AbstractVector{<:NamedTuple}) = begin 
    subsources = [
        ScaleHierarchy(source.log_scale, [centeredness])
        for centeredness in source.centeredness
    ]
    subdraws = [
        [
            (;draw.log_scale, weights=draw.weights[i:i])
            for draw in draws
        ]
        for i in eachindex(source.centeredness)
    ]
    subsources, subdraws
end
recombine(source::ScaleHierarchy, resources) = begin 
    ScaleHierarchy(
        source.log_scale, 
        vcat(getproperty.(resources, :centeredness)...)
    )
end


struct LocScaleHierarchy{I} <: AbstractReparametrizableDistribution
    info::I
end
LocScaleHierarchy(location, log_scale, c1, c2=c1) = LocScaleHierarchy((;location, log_scale, c1, c2))

lengths(source::LocScaleHierarchy) = (
    location=length(source.location), log_scale=length(source.log_scale), xic=length(source.c1)
)
reparametrization_parameters(source::LocScaleHierarchy) = map(finite_logit, (;source.c1, source.c2))
reparametrize(source::LocScaleHierarchy, parameters::NamedTuple) = LocScaleHierarchy(
    source.info.location, source.info.log_scale, map(logistic, parameters)...
)

lpdf_and_invariants(source::LocScaleHierarchy, draw::NamedTuple, lpdf=0.) = begin
    # Mirroring https://num.pyro.ai/en/stable/_modules/numpyro/infer/reparam.html#LocScaleReparam
    # delta = decentered_value - centered * fn.loc
    # value = fn.loc + jnp.power(fn.scale, 1 - centered) * delta
    weights = draw.location .+ xexpy.(
        draw.xic - source.c1 .* draw.location,
        draw.log_scale .* (1 .- source.c2)
    )
    # Mirroring https://num.pyro.ai/en/stable/_modules/numpyro/infer/reparam.html#LocScaleReparam
    # params["loc"] = fn.loc * centered
    # params["scale"] = fn.scale**centered
    prior_xic = Normal.(draw.location .* source.c1, exp.(draw.log_scale .* source.c2))
    if length(source.location) > 0
        lpdf += sum_logpdf(source.location, draw.location)
    end
    if length(source.log_scale) > 0
        lpdf += sum_logpdf(source.log_scale, draw.log_scale)
    end
    lpdf += sum_logpdf(prior_xic, draw.xic)
    (;lpdf, draw.location, draw.log_scale, weights)
end

lja_reparametrize(::LocScaleHierarchy, target::LocScaleHierarchy, invariants::NamedTuple, lja=0.) = begin 
    txic = xexpy.(
        invariants.weights .- invariants.location,
        invariants.log_scale .* (target.c2 .- 1)
    ) .+ target.c1 .* invariants.location
    # tdraw = StackedArray((;invariants.location, invariants.log_scale, xic=txic))
    tdraw = vcat(invariants.location, invariants.log_scale, txic)
    prior_txic = Normal.(invariants.location .* target.c1, exp.(invariants.log_scale .* target.c2))
    lja += sum_logpdf(prior_txic, txic)
    lja, tdraw
end

divide(source::LocScaleHierarchy, draws::AbstractVector{<:NamedTuple}) = begin 
    subsources = [
        LocScaleHierarchy(source.location, source.log_scale, [c1], [c2])
        for (c1, c2) in zip(source.c1, source.c2)
    ]
    subdraws = [
        [
            (;draw.location, draw.log_scale, weights=draw.weights[i:i])
            for draw in draws
        ]
        for i in eachindex(source.c1)
    ]
    subsources, subdraws
end
recombine(source::LocScaleHierarchy, resources) = begin 
    LocScaleHierarchy(
        source.location, source.log_scale, 
        vcat(getproperty.(resources, :c1)...), vcat(getproperty.(resources, :c2)...)
    )
end