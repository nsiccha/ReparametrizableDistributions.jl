# using TestEnv; TestEnv.activate("ReparametrizableDistributions");
using WarmupHMC, ReparametrizableDistributions, ReverseDiff, Distributions, Random, Test, Optim, ChainRulesTestUtils

import ReparametrizableDistributions: _logcdf, _invlogcdf

rmse(x,y; m=mean) = sqrt(m((x.-y).^2))
pairwise(f, arg, args...; kwargs...) = [
    f(lhs, rhs, args...; kwargs...) for lhs in arg, rhs in arg
]
transformation_tests(parametrizations, args...; kwargs...) = begin 
    @testset "identity" pairwise(reparametrization_test, parametrizations)
    @testset "rmse" pairwise(rmse_test, parametrizations, args...)
    @testset "loss" pairwise(loss_test, parametrizations, args...)
    @testset "easy_convergence" pairwise(easy_convergence_test, parametrizations, args...)
    @testset "hard_convergence" pairwise(hard_convergence_test, parametrizations, args...; kwargs...)
end 
test_draws(lhs; seed=0, n_draws=100) = randn(Xoshiro(seed), (length(lhs), n_draws))
reparametrization_test(lhs, rhs) = begin 
    parameters = WarmupHMC.reparametrization_parameters.([lhs, rhs])
    rlhs = WarmupHMC.reparametrize(lhs, parameters[2])
    lrlhs = WarmupHMC.reparametrize(rlhs, parameters[1])
    @test lhs == lrlhs
end
rmse_test(lhs, rhs, draws=test_draws(lhs), tol=1e-4) = begin
    rdraws = WarmupHMC.reparametrize(lhs, rhs, draws)
    lrdraws = WarmupHMC.reparametrize(rhs, lhs, rdraws)
    @test rmse(draws, lrdraws) < tol
end
loss_test(lhs, rhs, draws=test_draws(lhs)) = begin
    parameters = WarmupHMC.reparametrization_parameters.([lhs, rhs])
    loss = WarmupHMC.reparametrization_loss_function(lhs, draws)
    lloss, rloss = loss.(parameters)
    @test lloss <= rloss
end
easy_convergence_test(lhs, rhs, draws=test_draws(lhs)) = begin 
    flhs = WarmupHMC.find_reparametrization(:ReverseDiff, lhs, draws)
    lp, rp, fp = WarmupHMC.reparametrization_parameters.([lhs, rhs, flhs])
    @test rmse(lp, fp) <= rmse(rp, fp)
end
hard_convergence_test(lhs, rhs, draws=test_draws(lhs); kwargs...) = begin 
    rdraws = WarmupHMC.reparametrize(lhs, rhs, draws)
    flhs = WarmupHMC.find_reparametrization(:ReverseDiff, rhs, rdraws; kwargs...)
    lp, rp, fp = WarmupHMC.reparametrization_parameters.([lhs, rhs, flhs])
    @test rmse(lp, fp) <= rmse(rp, fp)
end

ChainRulesTestUtils.test_approx(actual::Distribution, expected::Distribution, args...; kwargs...) = test_approx(params(actual), params(expected), args...; kwargs...)

sensitivity_tests(dists, draws) = begin 
    broadcast(dists, draws) do dist, q
    # for dist in dists, q in draws
        lq = log(q)
        x = _invlogcdf(dist, lq)
        re_lq = _logcdf(dist, x)
        re_q = exp(re_lq)

        @test q ≈ re_q
        test_rrule(_logcdf, dist, x)
        test_rrule(_invlogcdf, dist, lq)
    end
end


rng = Xoshiro(0)
n_parameters = 4
n_draws = 100
# xi = randn(rng, (n_parameters, n_draws))
cs = [1, 10]#, 100]
scales = [0, 1]#, 2]

hierarchies = [
    ScaleHierarchy(Normal(), rand(rng, n_parameters))
    for i in 1:(length(cs)*length(scales))
]

concentrations = [
    c .* exp.(scale .* randn(rng, n_parameters))
    for c in cs
    for scale in scales
]
simplices = [
    GammaSimplex(Dirichlet(concentration))
    for concentration in concentrations
]


# @views WarmupHMC.lja_reparametrize(source::HSGP, target::HSGP, draw::AbstractVector, lja=0.) = begin  
#     sxic = draw[4:end]
#     lsds = HSGPs.log_sds(source, draw)
#     w = sxic .* exp.(lsds .* (1 .- source.centeredness))
#     txic = sxic .* exp.(lsds .* (target.centeredness .- source.centeredness))
#     trintercept = draw[1] + sum(w .* (target.mean_shift - source.mean_shift))
#     lja += -sum(lsds .* target.centeredness)
#     tdraw = vcat(trintercept, draw[2:3], txic)
#     lja, tdraw
# end

n_functions = 2
hsgps = [
    HSGP(0, 0, 0, ScaleHierarchy([], rand(rng, n_parameters)))
    for i in 1:12
]
# hsgp_xi = (randn(3+n_functions, n_draws))

r2d2s = [
    R2D2(0, 0, rand(rng, simplices), ScaleHierarchy([], rand(rng, n_parameters)))
    for concentration in concentrations
]

# @testset "New tests" begin 
#     transformation_tests(r2d2s)
# end

WarmupHMC.reparametrization_parameters(::Any) = Float64[]
@testset "All Tests" begin
    gammas = Gamma.(exp.(randn(rng, 100)), exp.(randn(rng, 100)))
    qs = rand(rng, 100)
    @testset "Sensitivities" begin
        # @testset "Gamma" sensitivity_tests(gammas, qs)
    end
    @testset "Transformation tests" begin
        @testset "ScaleHierarchy" transformation_tests(hierarchies)
        # @testset "GammaSimplex" transformation_tests(simplices)
        # @testset "HSGP" transformation_tests(hsgps; iterations=50)
        # @testset "R2D2" transformation_tests(r2d2s)
    end
end
# import ReparametrizableDistributions: info

# r2d2 = R2D2((
#     intercept=0, sigma_sq=1, R2=1, simplex=GammaSimplex(ones(5)), hierarchy=ones(5)
# ))
# info(r2d2, ones(13))