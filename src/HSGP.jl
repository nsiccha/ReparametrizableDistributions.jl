
abstract type AbstractHSGP <: AbstractReparametrizableDistribution end

struct HSGP{I} <: AbstractHSGP
    info::I
end

# https://github.com/avehtari/casestudies/blob/967cdb3a6432e8985886b96fda306645fe156a29/Motorcycle/gpbasisfun_functions.stan#L12-L14
HSGP(hyperprior::AbstractVector, x::AbstractVector, n_functions::Integer=32, boundary_factor::Real=1.5, centeredness=zeros(n_functions), mean_shift=zeros(n_functions)) = begin 
    idxs = 1:n_functions
    pre_eig = (-.25 * (pi/2/boundary_factor)^2) .* idxs .^ 2
    # sin(diag_post_multiply(rep_matrix(pi()/(2*L) * (x+L), M), linspaced_vector(M, 1, M)))/sqrt(L);
    X = sin.((x .+ boundary_factor) .* (pi/(2*boundary_factor)) .* idxs') ./ sqrt(boundary_factor)
    HSGP(hyperprior, pre_eig, X, centeredness, mean_shift)
end

logdensity_and_stuff(source::HSGP, draw::AbstractVector, lpdf=0.) = begin
    _info = info(source)
    _views = views(source, draw)
    # alpha * sqrt(sqrt(2*pi()) * rho) * exp(-0.25*(rho*pi()/2/L)^2 * linspaced_vector(M, 1, M)^2);
    lengthscale = exp.(_views.log_lengthscale)
    log_sds = (
        _views.log_sd + .25 * log(2*pi) + .5 * _views.log_lengthscale
    ) .+ lengthscale^2 .* _info.pre_eig
    weights = _views.xic .* exp.(log_sds .* (1 .- _info.centeredness))
    intercept = _views.effective_intercept - sum(weights .* _info.mean_shift)
    lpdf += logpdf.(_info.prior.intercept, intercept)
    lpdf += logpdf.(_info.prior.log_sd, _views.log_sd)
    lpdf += logpdf.(_info.prior.log_lengthscale, _views.log_lengthscale)
    prior_xic = Normal.(0., exp.(log_sds .* _info.centeredness))
    lpdf += sum(logpdf.(prior_xic, _views.xic))
    (;lpdf, lengthscale, log_sds, weights, intercept)
end
reparametrization_parameters(source::HSGP) = vcat(source.centeredness, source.mean_shift)
reparametrize(source::HSGP, parameters::AbstractVector) = HSGP(
    source.hyperprior,
    source.pre_eig,
    source.X,
    collect.(eachcol(reshape(parameters, (:, 2))))...
)

@views lja_reparametrize(source::HSGP, target::HSGP, draw::AbstractVector, lja=0.) = begin  
    sxic = draw[4:end]
    lsds = HSGPs.log_sds(source, draw)
    w = sxic .* exp.(lsds .* (1 .- source.centeredness))
    txic = sxic .* exp.(lsds .* (target.centeredness .- source.centeredness))
    trintercept = draw[1] + sum(w .* (target.mean_shift - source.mean_shift))
    lja += -sum(HSGPs.log_sds(source, draw) .* target.centeredness)
    tdraw = vcat(trintercept, draw[2:3], txic)
    lja, tdraw
end