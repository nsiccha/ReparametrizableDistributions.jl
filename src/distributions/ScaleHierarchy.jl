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
LocScaleHierarchy(location, log_scale, centeredness) = LocScaleHierarchy((;location, log_scale, centeredness))
length_info(source::LocScaleHierarchy) = Length((location=length(source.info.location), log_scale=length(source.info.log_scale), xic=length(source.info.centeredness)))
reparametrization_parameters(source::LocScaleHierarchy) = finite_logit.(source.info.centeredness)
reparametrize(source::LocScaleHierarchy, parameters::AbstractVector) = LocScaleHierarchy(source.info.location, source.info.log_scale, logistic.(parameters))

lpdf_and_invariants(source::LocScaleHierarchy, draw::NamedTuple, lpdf=0.) = begin
    _info = info(source)
    # Mirroring https://num.pyro.ai/en/stable/_modules/numpyro/infer/reparam.html#LocScaleReparam
    # delta = decentered_value - centered * fn.loc
    # value = fn.loc + jnp.power(fn.scale, 1 - centered) * delta
    weights = draw.location .+ xexpy.(
        draw.xic - _info.centeredness .* draw.location,
        draw.log_scale .* (1 .- _info.centeredness)
    )
    # Mirroring https://num.pyro.ai/en/stable/_modules/numpyro/infer/reparam.html#LocScaleReparam
    # params["loc"] = fn.loc * centered
    # params["scale"] = fn.scale**centered
    prior_xic = Normal.(draw.location .* _info.centeredness, exp.(draw.log_scale .* _info.centeredness))
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
        -invariants.log_scale .* (1 .- tinfo.centeredness)
    ) .+ tinfo.centeredness .* invariants.location
    tdraw = StackedVector((;invariants.location, invariants.log_scale, xic=txic))
    prior_txic = Normal.(invariants.location .* tinfo.centeredness, exp.(invariants.log_scale .* tinfo.centeredness))
    lja += sum_logpdf(prior_txic, txic)
    lja, tdraw
end