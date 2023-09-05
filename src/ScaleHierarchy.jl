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

logdensity_and_stuff(source::ScaleHierarchy, draw::AbstractVector, lpdf=0.) = begin
    _info, _views = info_and_views(source, draw)
    hierarchy = views.xic .* exp.(_views.log_sd .* (1 .- _info.centeredness))
    prior_xic = Normal.(0., exp.(_views.log_sd .* _info.centeredness))
    lpdf += sum_logpdf(_info.log_sd, _views.log_sd)
    lpdf += sum_logpdf(prior_xic, _views.xic)
    (;lpdf, hierarchy)
end

lja_reparametrize(source::ScaleHierarchy, target::ScaleHierarchy, draw::AbstractVector, lja=0.) = begin 
    _info, _views = info_and_views(source, draw)
    tinfo = info(target)
    txic = _views.xic .* exp.(_views.log_sd .* (tinfo.centeredness .- _info.centeredness))
    tdraw = vcat(_views.log_sd, txic)
    prior_txic = Normal.(0., exp.(_views.log_sd .* tinfo.centeredness))
    lja += sum_logpdf(prior_txic, txic)
    lja, tdraw
end