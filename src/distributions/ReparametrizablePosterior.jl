struct ReparametrizablePosterior{L,P} <: AbstractCompositeReparametrizableDistribution
    likelihood::L
    prior::P
end

remake(source::ReparametrizablePosterior, parts) = ReparametrizablePosterior(source.likelihood, parts)
parts(source::ReparametrizablePosterior) = source.prior

lpdf_and_invariants(source::ReparametrizablePosterior, draw::NamedTuple, lpdf=0.) = begin 
    prior_invariants = kmap(lpdf_and_invariants, source.prior, draw, lpdf)
    likelihood_invariants = source.likelihood(prior_invariants)
    lpdf += likelihood_invariants.lpdf + sum(getproperty.(values(prior_invariants), :lpdf))
    (;lpdf, likelihood=likelihood_invariants, prior_invariants...)
end

# lja_reparametrize(source::ReparametrizablePosterior, target::ReparametrizablePosterior, invariants::NamedTuple, lja=0.) = begin 
#     intermediates = values(kmap(lja_reparametrize, source.prior, target.prior, invariants.prior; lja=lja))
#     sum(first.(intermediates)), vcat(last.(intermediates)...)
# end

# find_reparametrization(source::ReparametrizablePosterior, draws; kwargs...) = begin
#     ReparametrizablePosterior(
#         source.likelihood,
#         kmap(find_reparametrization, source.prior, views(source, draws); kwargs...)
#     ) 
# end