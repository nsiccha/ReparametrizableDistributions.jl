struct MeanShift{I} <: AbstractReparametrizableDistribution
    info::I
end
MeanShift(intercept, mean_shift) = MeanShift((;intercept, mean_shift))

lengths(source::MeanShift) = (intercept=length(source.intercept),)
reparametrization_parameters(source::MeanShift) = source.mean_shift
reparametrize(source::MeanShift, parameters::AbstractVector) = MeanShift(source.intercept, parameters)

lpdf_and_invariants(source::MeanShift, draw::NamedTuple, lpdf=0.) = begin
    intercept = draw.intercept .+ sum(draw.weights .* source.mean_shift)
    lpdf += sum_logpdf(source.intercept, intercept)
    (;lpdf, intercept, draw.weights)
end
lja_reparametrize(::MeanShift, target::MeanShift, invariants::NamedTuple, lja=0.) = begin 
    (;lja, intercept=invariants.intercept .- sum(invariants.weights .* target.mean_shift))
end