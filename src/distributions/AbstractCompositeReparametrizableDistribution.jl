abstract type AbstractCompositeReparametrizableDistribution <: AbstractReparametrizableDistribution end

ACRD = AbstractCompositeReparametrizableDistribution

parts(::ACRD) = error("unimplemented")
parts(source::ACRD, target) = enure_like(parts(source), target)
remake(::ACRD, ::Any) = error("unimplemented")

lengths(source::ACRD) = map(length, parts(source))
reparametrization_parameters(source::ACRD) = map(reparametrization_parameters, parts(source))
reparametrize(source::T, parameters::NamedTuple) where {T<:ACRD} = remake(
    source, map(reparametrize, parts(source), parameters)
)

lpdf_and_invariants(::ACRD, ::NamedTuple, lpdf=0.) = error("unimplemented")

lja_reparametrize(source::T, target::T, invariants::NamedTuple, lja=0.) where {T<:ACRD} = begin 
    intermediates = kmap(lja_reparametrize, parts(source), parts(target), invariants, lja)
    (;lja=sum(getproperty.(values(intermediates), :lja)), intermediates...)
end

find_reparametrization(source::ACRD, draws; kwargs...) = begin
    return error("unimplemented")
    remake(
        source,
        kmap(find_reparametrization, parts(source), views(source, draws); kwargs...)
    ) 
end