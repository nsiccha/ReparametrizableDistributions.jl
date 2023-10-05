
abstract type AbstractHSGP <: AbstractReparametrizableDistribution end

struct HSGP{I} <: AbstractHSGP
    info::I
end
HSGP(intercept, log_sd, log_lengthscale; intercept_shift, centeredness, kwargs...) = HSGP(
    MeanShift(intercept, intercept_shift),
    log_sd, log_lengthscale,
    ScaleHierarchy([], centeredness); kwargs...
)
HSGP(intercept, log_sd, log_lengthscale, hierarchy; kwargs...) = HSGP(
    (;intercept, log_sd, log_lengthscale, hierarchy, hsgp_extra(;n_functions=length(hierarchy), kwargs...)...)
)
hsgp_extra(;x, n_functions::Integer=32, boundary_factor::Real=1.5) = begin 
    idxs = 1:n_functions
    # sin(diag_post_multiply(rep_matrix(pi()/(2*L) * (x+L), M), linspaced_vector(M, 1, M)))/sqrt(L);
    X = sin.((x .+ boundary_factor) .* (pi/(2*boundary_factor)) .* idxs') ./ sqrt(boundary_factor)
    # alpha * sqrt(sqrt(2*pi()) * rho) * exp(-0.25*(rho*pi()/2/L)^2 * linspaced_vector(M, 1, M)^2);
    pre_eig = (-.25 * (pi/2/boundary_factor)^2) .* idxs .^ 2
    (;X, pre_eig)
end
length_info(source::HSGP) = Length((intercept=1, log_sd=1, log_lengthscale=1, hierarchy=length(source.info.hierarchy)))
reparametrize(source::HSGP, parameters::NamedTuple) = HSGP(map(reparametrize, info(source), parameters))


lpdf_and_invariants(source::HSGP, draw::NamedTuple, lpdf=0.) = begin
    _info = info(source)
    # alpha * sqrt(sqrt(2*pi()) * rho) * exp(-0.25*(rho*pi()/2/L)^2 * linspaced_vector(M, 1, M)^2);
    lengthscale = exp.(draw.log_lengthscale)
    log_scale = (
        draw.log_sd .+ .25 * log(2*pi) .+ .5 * draw.log_lengthscale
    ) .+ lengthscale.^2 .* _info.pre_eig
    hierarchy = lpdf_and_invariants(_info.hierarchy, (;log_scale, xic=draw.hierarchy))
    intercept = lpdf_and_invariants(_info.intercept, (;draw.intercept, hierarchy.weights))
    lpdf += intercept.lpdf
    lpdf += sum_logpdf(_info.log_sd, draw.log_sd)
    lpdf += sum_logpdf(_info.log_lengthscale, draw.log_lengthscale)
    lpdf += hierarchy.lpdf
    y = intercept.intercept .+ source.X * hierarchy.weights
    (;lpdf, intercept, draw.log_sd, draw.log_lengthscale, hierarchy, y)
end
lja_reparametrize(source::HSGP, target::HSGP, invariants::NamedTuple, lja=0.) = begin  
    _info = info(source)
    tinfo = info(target)
    lja_intercept, tdraw_intercept = lja_reparametrize(_info.intercept, tinfo.intercept, invariants.intercept)
    lja_hierarchy, tdraw_hierarchy = lja_reparametrize(_info.hierarchy, tinfo.hierarchy, invariants.hierarchy)
    lja += lja_intercept
    lja += lja_hierarchy
    tdraw = vcat(
        views(tinfo.intercept, tdraw_intercept).intercept, 
        invariants.log_sd, invariants.log_lengthscale, 
        views(tinfo.hierarchy, tdraw_hierarchy).xic
    )
    lja, tdraw
end

divide(source::HSGP, draws::AbstractVector{<:NamedTuple}) = begin 
    subsources = (source.intercept, source.hierarchy)
    subdraws = getproperty.(draws, :intercept), getproperty.(draws, :hierarchy)
    subsources, subdraws
end
recombine(source::HSGP, resources) = begin 
    HSGP(merge(source.info, (intercept=resources[1], hierarchy=resources[2])))
end