struct ReparametrizablePosterior{L,P} <: AbstractReparametrizableDistribution
    likelihood::L
    prior::P
end

info(source::ReparametrizablePosterior) = source.prior
reparametrize(source::ReparametrizablePosterior, parameters::NamedTuple) = ReparametrizablePosterior(
    source.likelihood, map(reparametrize, source.prior, parameters)
)

lpdf_and_invariants(source::ReparametrizablePosterior, draw::NamedTuple, lpdf=0.) = begin 
    prior_invariants = map(lpdf_and_invariants, source.prior, draw)
    likelihood_invariants = source.likelihood(prior_invariants)
    lpdf += likelihood_invariants.lpdf + sum(getproperty.(values(prior_invariants), :lpdf))
    (;lpdf, prior=prior_invariants, likelihood=likelihood_invariants)
end

lja_reparametrize(source::ReparametrizablePosterior, target::ReparametrizablePosterior, invariants::NamedTuple, lja=0.) = begin 
    intermediates = values(map(lja_reparametrize, source.prior, target.prior, invariants.prior))
    sum(first.(intermediates)), vcat(last.(intermediates)...)
end

find_reparametrization(source::ReparametrizablePosterior, draws::AbstractMatrix; kwargs...) = begin
    ReparametrizablePosterior(
        source.likelihood,
        map((args...)->find_reparametrization(args...; kwargs...), source.prior, views(source, draws))
    ) 
end