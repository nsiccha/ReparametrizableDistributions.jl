abstract type AbstractCompositeReparametrizableDistribution <: AbstractReparametrizableDistribution end
ACRD = AbstractCompositeReparametrizableDistribution
# parts(source::ACRD, target) = ensure_like(parts(source), target)

reparametrization_parameters(source::ACRD) = map(reparametrization_parameters, parts(source))
optimization_parameters_fn(source::ACRD) = map(optimization_parameters_fn, parts(source))
reparametrize(source::ACRD, parameters::NamedTuple) = recombine(
    source, map(reparametrize, parts(source), parameters)
)

# IMPLEMENT THIS
lpdf_update(::ACRD, ::NamedTuple, lpdf=0.) = error("unimplemented")
lja_update(source::ACRD, target::ACRD, invariants::NamedTuple, lja=0.) = begin 
    intermediates = kmap(lja_reparametrize, parts(source), parts(target), invariants, lja)
    (;lja=sum(getproperty.(values(intermediates), :lja)), intermediates...)
end


divide(source::ACRD, draws::AbstractVector{<:NamedTuple}) = parts(source), (;
    ((key, getproperty.(draws, key)) for key in keys(parts(source)))...
)
# IMPLEMENT THIS
recombine(::ACRD, ::Any) = error("unimplemented")