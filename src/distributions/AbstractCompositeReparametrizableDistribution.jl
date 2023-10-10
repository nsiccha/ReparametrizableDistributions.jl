abstract type AbstractCompositeReparametrizableDistribution <: AbstractReparametrizableDistribution end
ACRD = AbstractCompositeReparametrizableDistribution
# parts(source::ACRD, target) = ensure_like(parts(source), target)

reparametrization_parameters(source::ACRD) = map(reparametrization_parameters, parts(source))
optimization_parameters_fn(source::ACRD) = map(optimization_parameters_fn, parts(source))
reparametrize(source::ACRD, parameters::NamedTuple) = remake(
    source, map(reparametrize, parts(source), parameters)
)
# IMPLEMENT THIS
remake(::ACRD, ::Any) = error("unimplemented")

# IMPLEMENT THIS
lpdf_update(::ACRD, ::NamedTuple, lpdf=0.) = error("unimplemented")
lja_update(source::ACRD, target::ACRD, invariants::NamedTuple, lja=0.) = begin 
    intermediates = kmap(lja_reparametrize, parts(source), parts(target), invariants, lja)
    (;lja=sum(getproperty.(values(intermediates), :lja)), intermediates...)
end


divide(source::ACRD, draws::AbstractVector{<:NamedTuple}) = parts(source)
# find_reparametrization(source::ACRD, draws::AbstractMatrix; kwargs...) = begin
#     return error("unimplemented")
#     remake(
#         source,
#         kmap(find_reparametrization, parts(source), views(source, draws); kwargs...)
#     ) 
# end
