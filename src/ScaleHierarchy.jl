struct ScaleHierarchy{I} <: AbstractReparametrizableDistribution
    info::I
end

ScaleHierarchy(log_sd, centeredness) = ScaleHierarchy((;log_sd, centeredness))
Base.length(source::ScaleHierarchy) = sum(length.(values(info(source))))
views(source::ScaleHierarchy, draw::AbstractVector) = (
    _info = info(source);
    views((;_info.log_sd, xic=_info.centeredness), draw)
)
reparametrization_parameters(source::ScaleHierarchy) = logit.(info(source).centeredness)
reparametrize(source::ScaleHierarchy, parameters::AbstractVector) = ScaleHierarchy(info(source).log_sd, logistic.(parameters))

lpdf_and_invariants(source::ScaleHierarchy, draw::NamedTuple, lpdf=0.) = begin
    _info = info(source)
    hierarchy = views.xic .* exp.(draw.log_sd .* (1 .- _info.centeredness))
    prior_xic = Normal.(0., exp.(draw.log_sd .* _info.centeredness))
    lpdf += sum_logpdf(_info.log_sd, draw.log_sd)
    lpdf += sum_logpdf(prior_xic, draw.xic)
    (;lpdf, draw.log_sd, hierarchy)
end

lja_reparametrize(source::ScaleHierarchy, target::ScaleHierarchy, draw::NamedTuple, lja=0.) = begin 
    _info = info(source)
    tinfo = info(target)
    txic = draw.xic .* exp.(draw.log_sd .* (tinfo.centeredness .- _info.centeredness))
    tdraw = vcat(draw.log_sd, txic)
    prior_txic = Normal.(0., exp.(draw.log_sd .* tinfo.centeredness))
    lja += sum_logpdf(prior_txic, txic)
    lja, tdraw
end