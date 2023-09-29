struct ScaleHierarchy{I} <: AbstractReparametrizableDistribution
    info::I
end

finite_logit(x, reg=1e-4) = logit(.5 + (x-.5)*(1-reg))

ScaleHierarchy(log_scale, centeredness) = ScaleHierarchy((;log_scale, centeredness))
length_info(source::ScaleHierarchy) = Length((log_scale=length(source.info.log_scale), xic=length(source.info.centeredness)))
reparametrization_parameters(source::ScaleHierarchy) = finite_logit.(source.info.centeredness)
reparametrize(source::ScaleHierarchy, parameters::AbstractVector) = ScaleHierarchy(source.info.log_scale, logistic.(parameters))

lpdf_and_invariants(source::ScaleHierarchy, draw::NamedTuple, lpdf=0.) = begin
    _info = info(source)
    weights = xexpy.(draw.xic, draw.log_scale .* (1 .- _info.centeredness))
    # xi = xexpy.(draw.xic, draw.log_scale .* (0 .- _info.centeredness))
    prior_xic = Normal.(0., exp.(draw.log_scale .* _info.centeredness))
    if length(_info.log_scale) > 0
        lpdf += sum_logpdf(_info.log_scale, draw.log_scale)
    end
    lpdf += sum_logpdf(prior_xic, draw.xic)
    (;lpdf, draw.log_scale, weights)#, xi)
end

lja_reparametrize(::ScaleHierarchy, target::ScaleHierarchy, invariants::NamedTuple, lja=0.) = begin 
    # _info = info(source)
    tinfo = info(target)
    txic = xexpy.(invariants.weights, invariants.log_scale .* (tinfo.centeredness .- 1))
    # txic = xexpy.(invariants.xi, invariants.log_scale .* (tinfo.centeredness .- 0))
    tdraw = StackedVector((;invariants.log_scale, xic=txic))
    prior_txic = Normal.(0., exp.(invariants.log_scale .* tinfo.centeredness))
    lja += sum_logpdf(prior_txic, txic)
    lja, tdraw
end


struct LocScaleHierarchy{I} <: AbstractReparametrizableDistribution
    info::I
end
# LocScaleHierarchy(location, log_scale, centeredness) = LocScaleHierarchy((;location, log_scale, centeredness))
LocScaleHierarchy(location, log_scale, c1, c2=c1) = LocScaleHierarchy((;location, log_scale, c1, c2))
length_info(source::LocScaleHierarchy) = Length((location=length(source.info.location), log_scale=length(source.info.log_scale), xic=length(source.info.c1)))
reparametrization_parameters(source::LocScaleHierarchy) = finite_logit.(vcat(source.info.c1, source.info.c2))
reparametrize(source::LocScaleHierarchy, parameters::AbstractVector) = LocScaleHierarchy(source.info.location, source.info.log_scale, eachcol(reshape(logistic.(parameters), (:, 2)))...)

lpdf_and_invariants(source::LocScaleHierarchy, draw::NamedTuple, lpdf=0.) = begin
    _info = info(source)
    # Mirroring https://num.pyro.ai/en/stable/_modules/numpyro/infer/reparam.html#LocScaleReparam
    # delta = decentered_value - centered * fn.loc
    # value = fn.loc + jnp.power(fn.scale, 1 - centered) * delta
    weights = draw.location .+ xexpy.(
        draw.xic - _info.c1 .* draw.location,
        draw.log_scale .* (1 .- _info.c2)
    )
    # Mirroring https://num.pyro.ai/en/stable/_modules/numpyro/infer/reparam.html#LocScaleReparam
    # params["loc"] = fn.loc * centered
    # params["scale"] = fn.scale**centered
    prior_xic = Normal.(draw.location .* _info.c1, exp.(draw.log_scale .* _info.c2))
    if length(_info.location) > 0
        lpdf += sum_logpdf(_info.location, draw.location)
    end
    if length(_info.log_scale) > 0
        lpdf += sum_logpdf(_info.log_scale, draw.log_scale)
    end
    lpdf += sum_logpdf(prior_xic, draw.xic)
    (;lpdf, draw.location, draw.log_scale, weights)
end

lja_reparametrize(::LocScaleHierarchy, target::LocScaleHierarchy, invariants::NamedTuple, lja=0.) = begin 
    # _info = info(source)
    tinfo = info(target)
    txic = xexpy.(
        invariants.weights .- invariants.location,
        invariants.log_scale .* (tinfo.c2 .- 1)
    ) .+ tinfo.c1 .* invariants.location
    tdraw = StackedVector((;invariants.location, invariants.log_scale, xic=txic))
    prior_txic = Normal.(invariants.location .* tinfo.c1, exp.(invariants.log_scale .* tinfo.c2))
    lja += sum_logpdf(prior_txic, txic)
    lja, tdraw
end

divide(source::LocScaleHierarchy, draws::AbstractMatrix) = begin 
    subsources = [
        LocScaleHierarchy(source.location, source.log_scale, [c1], [c2])
        for (c1, c2) in zip(source.c1, source.c1)
    ]
    subdraws = [
        vcat(draws[1:2,:], row') for row in eachrow(draws[3:end, :])
    ]
    subsources, subdraws
end
recombine(sources::Vector{<:LocScaleHierarchy}) = begin 
    LocScaleHierarchy(
        sources[1].location, sources[1].log_scale, 
        vcat(getproperty.(sources, :c1)), vcat(getproperty.(sources, :c2))
    )
end