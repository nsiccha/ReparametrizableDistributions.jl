struct ScaleHierarchy{I} <: AbstractReparametrizableDistribution
    info::I
end

ScaleHierarchy(log_scale, centeredness) = ScaleHierarchy((;log_scale, centeredness))
length_info(source::ScaleHierarchy) = Length((log_scale=length(source.info.log_scale), xic=length(source.info.centeredness)))
reparametrization_parameters(source::ScaleHierarchy) = logit.(source.info.centeredness)
reparametrize(source::ScaleHierarchy, parameters::AbstractVector) = ScaleHierarchy(source.info.log_scale, logistic.(parameters))

lpdf_and_invariants(source::ScaleHierarchy, draw::NamedTuple, lpdf=0.) = begin
    _info = info(source)
    weights = xexpy.(draw.xic, draw.log_scale .* (1 .- _info.centeredness))
    prior_xic = Normal.(0., exp.(draw.log_scale .* _info.centeredness))
    if length(_info.log_scale) > 0
        lpdf += sum_logpdf(_info.log_scale, draw.log_scale)
    end
    lpdf += sum_logpdf(prior_xic, draw.xic)
    (;lpdf, draw.log_scale, weights)
end

lja_reparametrize(source::ScaleHierarchy, target::ScaleHierarchy, invariants::NamedTuple, lja=0.) = begin 
    # _info = info(source)
    tinfo = info(target)
    txic = xexpy.(invariants.weights, invariants.log_scale .* (tinfo.centeredness .- 1))
    # tdraw = vcat(invariants.log_scale, txic)
    tdraw = StackedVector((;invariants.log_scale, xic=txic))
    prior_txic = Normal.(0., exp.(invariants.log_scale .* tinfo.centeredness))
    lja += sum_logpdf(prior_txic, txic)
    lja, tdraw
end