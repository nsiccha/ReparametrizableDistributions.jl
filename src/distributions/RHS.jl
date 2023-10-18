struct RHS{I} <: AbstractCompositeReparametrizableDistribution
    info::I
end
RHS(nu_global, nu_local, slab_scale, slab_df, scale_global, centeredness) = RHS((;
    log_c=.5log_transform(slab_scale^2 * InverseGamma(.5slab_df, .5slab_df)),
    log_lambda=fill(log_transform(TDist(nu_local)), size(centeredness)),
    log_tau=log_transform(2*scale_global * TDist(nu_global)),
    hierarchy=ScaleHierarchy((), centeredness)
))
parts(source::RHS) = source.info

lpdf_update(source::RHS, draw::NamedTuple, lpdf=0.) = begin
    # https://github.com/avehtari/casestudies/blob/967cdb3a6432e8985886b96fda306645fe156a29/Birthdays/gpbf8rhs.stan#L87-L91
    #   real c_f4 = slab_scale * sqrt(caux_f4); // slab scale
    #   beta ~ normal(0, sqrt( c^2 * square(lambda) ./ (c^2 + tau^2*square(lambda)))*tau);
    # scale = c * lambda ./ sqrt(c^2 + tau^2*square(lambda)))*tau
    #   lambda ~ student_t(nu_local, 0, 1);
    #   tau ~ student_t(nu_global, 0, scale_global*2);
    #   caux ~ inv_gamma(0.5*slab_df, 0.5*slab_df);
    log_c, log_lambda, log_tau = draw.log_c, draw.log_lambda, draw.log_tau

    log_scale = log_c .+ log_lambda .- .5 .* logaddexp.(2 .* log_c, 2 .* (log_tau .+ log_lambda)) .+ log_tau;
    hierarchy = lpdf_and_invariants(source.hierarchy, (;log_scale, weights=draw.hierarchy))
    lpdf += sum_logpdf(source.log_c, draw.log_c)
    lpdf += sum_logpdf(source.log_lambda, draw.log_lambda)
    lpdf += sum_logpdf(source.log_tau, draw.log_tau)
    lpdf += hierarchy.lpdf
    (;lpdf, hierarchy, weights=hierarchy.weights)
end