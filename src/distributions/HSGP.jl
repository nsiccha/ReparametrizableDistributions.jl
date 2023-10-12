
# abstract type AbstractHSGP <: AbstractReparametrizableDistribution end

struct HSGP{I} <: AbstractCompositeReparametrizableDistribution
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
    # alpha * sqrt(sqrt(2*pi()) * rho) * exp(-0.25*(rho*pi()/2/L)^2 * linspaced_vector(M, 1, M)^2);
    lengthscale = exp.(draw.log_lengthscale)
    log_scale = (
        draw.log_sd .+ .25 * log(2*pi) .+ .5 * draw.log_lengthscale
    ) .+ lengthscale.^2 .* source.pre_eig
    log_scale = logaddexp.(1e-8, log_scale)
    hierarchy = lpdf_and_invariants(source.hierarchy, (;log_scale, weights=draw.hierarchy), lpdf)
    intercept = lpdf_and_invariants(source.intercept, (;draw.intercept, hierarchy.weights), lpdf)
    lpdf += intercept.lpdf
    lpdf += sum_logpdf(source.log_sd, draw.log_sd)
    lpdf += sum_logpdf(source.log_lengthscale, draw.log_lengthscale)
    lpdf += hierarchy.lpdf
    y = intercept.intercept .+ source.X * hierarchy.weights
    (;lpdf, intercept, hierarchy, y)
end
recombine(source::HSGP, reparts::NamedTuple) = HSGP(merge(info(source), reparts))



struct PHSGP{I} <: AbstractCompositeReparametrizableDistribution
    info::I
end
PHSGP(log_sd, log_lengthscale; centeredness, kwargs...) = PHSGP(
    log_sd, log_lengthscale,
    ScaleHierarchy([], centeredness); kwargs...
)
PHSGP(log_sd, log_lengthscale, hierarchy; kwargs...) = PHSGP(
    (;log_sd, log_lengthscale, hierarchy, phsgp_extra(;n_functions=length(hierarchy), kwargs...)...)
)
phsgp_extra(;x, n_functions::Integer=32, boundary_factor::Real=1.5) = begin 
    idxs = 1:(n_functions ÷ 2)
    # return append_col(
    #     cos(diag_post_multiply(rep_matrix(2*pi()*x/L, M/2), linspaced_vector(M/2, 1, M/2))),
    #     sin(diag_post_multiply(rep_matrix(2*pi()*x/L, M/2), linspaced_vector(M/2, 1, M/2)))
    # );
    xi = (2 .* pi .* x ./ boundary_factor) .* idxs'
    X = hcat(cos.(xi), sin.(xi))
    (;X, vcat(idxs, idxs))
end
parts(source::PHSGP) = (;source.log_sd, source.log_lengthscale, source.hierarchy)

lpdf_update(source::PHSGP, draw::NamedTuple, lpdf=0.) = begin
    # real a = exp(-2*log_lengthscale);
    # vector[M/2] q = log_sd + 0.5 * (log(2) - a + to_vector(log_modified_bessel_first_kind(linspaced_int_array(M/2, 1, M/2), a)));
    # return append_row(q,q);
    a = exp.(-2 .* draw.log_lengthscale)
    log_scale = (
        # Let's see whether this is stable
        draw.log_sd .+ .5 * (log(2) .+ log.(besselix.(source.idxs, a)))
    )
    log_scale = logaddexp.(1e-8, log_scale)
    hierarchy = lpdf_and_invariants(source.hierarchy, (;log_scale, weights=draw.hierarchy), lpdf)
    lpdf += sum_logpdf(source.log_sd, draw.log_sd)
    lpdf += sum_logpdf(source.log_lengthscale, draw.log_lengthscale)
    lpdf += hierarchy.lpdf
    y = source.X * hierarchy.weights
    (;lpdf, hierarchy, y)
end
recombine(source::PHSGP, reparts::NamedTuple) = PHSGP(merge(info(source), reparts))