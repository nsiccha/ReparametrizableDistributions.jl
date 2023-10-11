struct ReparametrizablePosterior{L,P} <: AbstractCompositeReparametrizableDistribution
    likelihood::L
    prior::P
end
parts(source::ReparametrizablePosterior) = source.prior
recombine(source::ReparametrizablePosterior, prior) = ReparametrizablePosterior(
    source.likelihood, prior
)
divide(source::ReparametrizablePosterior, draws::AbstractMatrix) = parts(source), to_nt(source, draws)

lpdf_and_invariants(source::ReparametrizablePosterior, draw::NamedTuple, lpdf=0.) = begin 
    prior_invariants = kmap(lpdf_and_invariants, source.prior, draw, lpdf)
    if isa(lpdf, Ignore)
        (;lpdf, prior_invariants...)
    else
        likelihood_invariants = source.likelihood(prior_invariants)
        lpdf += likelihood_invariants.lpdf + sum(getproperty.(values(prior_invariants), :lpdf))
        (;lpdf, likelihood=likelihood_invariants, prior_invariants...)
    end
end