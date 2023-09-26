struct MeanShift{I} <: AbstractReparametrizableDistribution
    info::I
end
MeanShift(intercept, mean_shift) = MeanShift((;intercept, mean_shift))
length_info(source::MeanShift) = Length((intercept=length(source.info.intercept), weights=length(source.info.mean_shift)))
reparametrization_parameters(source::MeanShift) = source.info.mean_shift
reparametrize(source::MeanShift, parameters::AbstractVector) = MeanShift(source.info.intercept, parameters)

lpdf_and_invariants(source::MeanShift, draw::NamedTuple, lpdf=0.) = begin
    _info = info(source)
    intercept = draw.intercept .- sum(draw.weights .* _info.mean_shift)
    lpdf += sum_logpdf(_info.intercept, intercept)
    (;lpdf, intercept, draw.weights)
end

lja_reparametrize(::MeanShift, target::MeanShift, invariants::NamedTuple, lja=0.) = begin 
    tinfo = info(target)
    tintercept = invariants.intercept .+ sum(invariants.weights .* tinfo.mean_shift)
    tdraw = StackedVector((;intercept=tintercept, weights=invariants.weights))
    lja, tdraw
end