
# abstract type AbstractHSGP <: AbstractReparametrizableDistribution end

struct HSGP{I} <: AbstractReparametrizableDistribution
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
parts(source::HSGP) = (;source.intercept, source.log_sd, source.log_lengthscale, source.hierarchy)


lpdf_update(source::HSGP, draw::NamedTuple, lpdf=0.) = begin
    _info = info(source)
    # alpha * sqrt(sqrt(2*pi()) * rho) * exp(-0.25*(rho*pi()/2/L)^2 * linspaced_vector(M, 1, M)^2);
    lengthscale = exp.(draw.log_lengthscale)
    log_scale = (
        draw.log_sd .+ .25 * log(2*pi) .+ .5 * draw.log_lengthscale
    ) .+ lengthscale.^2 .* _info.pre_eig
    log_scale = logaddexp.(1e-8, log_scale)
    hierarchy = lpdf_and_invariants(_info.hierarchy, (;log_scale, xic=draw.hierarchy), lpdf)
    intercept = lpdf_and_invariants(_info.intercept, (;draw.intercept, hierarchy.weights), lpdf)
    lpdf += intercept.lpdf
    lpdf += sum_logpdf(_info.log_sd, draw.log_sd)
    lpdf += sum_logpdf(_info.log_lengthscale, draw.log_lengthscale)
    lpdf += hierarchy.lpdf
    y = intercept.intercept .+ source.X * hierarchy.weights
    (;lpdf, intercept, hierarchy, y)
end