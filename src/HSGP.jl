
abstract type AbstractHSGP <: AbstractReparametrizableDistribution end

struct HSGP{I} <: AbstractHSGP
    info::I
end
HSGP(intercept, log_sd, log_lengthscale, hierarchy; kwargs...) = HSGP(
    (;intercept, log_sd, log_lengthscale, hierarchy, hsgp_extra(;n_functions=length(hierarchy), kwargs...)...)
)
hsgp_extra(;n_functions::Integer=32, boundary_factor::Real=1.5) = begin 
    idxs = 1:n_functions
    # alpha * sqrt(sqrt(2*pi()) * rho) * exp(-0.25*(rho*pi()/2/L)^2 * linspaced_vector(M, 1, M)^2);
    pre_eig = (-.25 * (pi/2/boundary_factor)^2) .* idxs .^ 2
    (;pre_eig, mean_shift=zeros(n_functions))
end
length_info(source::HSGP) = Length((intercept=1, log_sd=1, log_lengthscale=1, hierarchy=length(source.info.hierarchy)))

# # https://github.com/avehtari/casestudies/blob/967cdb3a6432e8985886b96fda306645fe156a29/Motorcycle/gpbasisfun_functions.stan#L12-L14
# HSGP(hyperprior::AbstractVector, x::AbstractVector, n_functions::Integer=32, boundary_factor::Real=1.5, centeredness=zeros(n_functions), mean_shift=zeros(n_functions)) = begin 
#     # sin(diag_post_multiply(rep_matrix(pi()/(2*L) * (x+L), M), linspaced_vector(M, 1, M)))/sqrt(L);
#     X = sin.((x .+ boundary_factor) .* (pi/(2*boundary_factor)) .* idxs') ./ sqrt(boundary_factor)
#     HSGP(hyperprior, pre_eig, X, centeredness, mean_shift)
# end
# reparametrization_parameters(source::HSGP) = vcat(source.centeredness, source.mean_shift)
# reparametrize(source::HSGP, parameters::AbstractVector) = HSGP(
#     source.hyperprior,
#     source.pre_eig,
#     source.X,
#     collect.(eachcol(reshape(parameters, (:, 2))))...
# )

lpdf_and_invariants(source::HSGP, draw::NamedTuple, lpdf=0.) = begin
    _info = info(source)
    # alpha * sqrt(sqrt(2*pi()) * rho) * exp(-0.25*(rho*pi()/2/L)^2 * linspaced_vector(M, 1, M)^2);
    lengthscale = exp.(draw.log_lengthscale)
    log_scale = (
        draw.log_sd .+ .25 * log(2*pi) .+ .5 * draw.log_lengthscale
    ) .+ lengthscale.^2 .* _info.pre_eig
    hierarchy = lpdf_and_invariants(_info.hierarchy, (;log_scale, xic=draw.hierarchy))
    intercept = lpdf_and_invariants(_info.intercept, (;draw.intercept, hierarchy.weights))
    # intercept = draw.intercept .- sum(hierarchy.weights .* _info.mean_shift)
    lpdf += intercept.lpdf
    # lpdf += sum_logpdf(_info.intercept, intercept)
    lpdf += sum_logpdf(_info.log_sd, draw.log_sd)
    lpdf += sum_logpdf(_info.log_lengthscale, draw.log_lengthscale)
    lpdf += hierarchy.lpdf
    (;lpdf, intercept, draw.log_sd, draw.log_lengthscale, hierarchy)
end
@views lja_reparametrize(source::HSGP, target::HSGP, invariants::NamedTuple, lja=0.) = begin  
    _info = info(source)
    tinfo = info(target)
    lja_intercept, tdraw_intercept = lja_reparametrize(_info.intercept, tinfo.intercept, invariants.intercept)
    lja_hierarchy, tdraw_hierarchy = lja_reparametrize(_info.hierarchy, tinfo.hierarchy, invariants.hierarchy)
    lja += lja_intercept
    lja += lja_hierarchy
    tdraw = vcat(views(tdraw_intercept).intercept, invariants.log_sd, invariants.log_lengthscale, views(tdraw_hierarchy).xic)
    if sum(isnan.(tdraw))> 0
        println(invariants)
        println(tinfo)
        println((;views(tdraw_intercept).intercept, invariants.log_sd, invariants.log_lengthscale, views(tdraw_hierarchy).xic))
        # println(views(tdraw_hierarchy))
        # println((tdraw, isnan.(tdraw)))
    end 
    lja, tdraw
end
