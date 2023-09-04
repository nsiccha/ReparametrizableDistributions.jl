
# defaults(::AbstractReparametrizableDistribution) = NamedTuple()
# info(source::AbstractReparametrizableDistribution) = merge(defaults(source), source.info)
# info(source::AbstractReparametrizableDistribution, ::AbstractVector) = info(source)
# # Base.length(source::AbstractReparametrizableDistribution) = sum(length.(values(info(source))))
# reparametrization_parameters(source::AbstractReparametrizableDistribution) = map(reparametrization_parameters, info(source))
# reparametrize(source::T, parameters::AbstractVector) where {T <: AbstractReparametrizableDistribution} = T(map(
#     reparametrize, info(source), views(reparametrization_parameters(source), parameters)
# ))

# struct LocScaleHierarchy{I} <: AbstractReparametrizableDistribution
#     info::I
# end
# defaults(::LocScaleHierarchy) = (location=0., )
# info(source::LocScaleHierarchy, draw::AbstractVector) = begin 
#     _info = info(source)
#     # _views = views(source, draw)
#     # lja = sum(exp.(_info.centeredness))
#     # lpdf = sum(logpdf.(_info.distributions, draw))
#     # invariants = 
#     (;lpdf, invariants, reparametrization)
# end