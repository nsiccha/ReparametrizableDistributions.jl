# using TestEnv; TestEnv.activate("ReparametrizableDistributions");
using WarmupHMC, ReparametrizableDistributions, ReverseDiff, Distributions, Random, Test, Optim, ChainRulesTestUtils, NaNStatistics, Plots

import ReparametrizableDistributions: _logcdf, _invlogcdf

rmse(x,y; m=nanmean) = sqrt(m((x.-y).^2))
pairwise(f, arg, args...; kwargs...) = [
    f(lhs, rhs, args...; kwargs...) for lhs in arg, rhs in arg
]
transformation_tests(parametrizations, args...; kwargs...) = begin 
    @testset "reparametrization" pairwise(reparametrization_test, parametrizations)
    @testset "nan" pairwise(nan_test, parametrizations, args...)
    @testset "rmse" pairwise(rmse_test, parametrizations, args...)
    @testset "loss" pairwise(loss_test, parametrizations, args...)
    @testset "easy_convergence" pairwise(easy_convergence_test, parametrizations, args...)
    @testset "hard_convergence" pairwise(hard_convergence_test, parametrizations, args...; kwargs...)
end 
test_draws(lhs; seed=0, n_draws=100) = randn(Xoshiro(seed), (length(lhs), n_draws))
reparametrization_test(lhs, rhs, tol=1e-4) = begin 
    parameters = WarmupHMC.reparametrization_parameters.([lhs, rhs])
    rlhs = WarmupHMC.reparametrize(lhs, parameters[2])
    lrlhs = WarmupHMC.reparametrize(rlhs, parameters[1])
    @test rmse(parameters[2], WarmupHMC.reparametrization_parameters(rlhs)) <= tol
    @test rmse(parameters[1], WarmupHMC.reparametrization_parameters(lrlhs)) <= tol
end
count_nan(x) = sum(map(count_nan, x))
count_nan(x::Real) = isnan(x) ? 1 : 0 
nan_test(lhs, rhs, draws=test_draws(lhs), tol=1e-4) = begin
    rdraws = WarmupHMC.reparametrize(lhs, rhs, draws)
    @test count_nan(WarmupHMC.lpdf_and_invariants.([lhs], eachcol(draws))) == 0
    @test count_nan(rdraws) == 0
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
n_parametrizations = 12
n_parameters = 4
n_draws = 100
# xi = randn(rng, (n_parameters, n_draws))
cs = exp.(randn(rng, n_parametrizations))
scales = exp.(randn(rng, n_parametrizations))

concentrations = [
    c .* exp.(scale .* randn(rng, n_parameters))
    for (c, scale) in zip(cs, scales)
]

hierarchies = [
    ScaleHierarchy(Normal(), rand(rng, n_parameters))
    for concentration in concentrations
]
mean_shifts = [
    MeanShift(Normal(), randn(rng, n_parameters))
    for concentration in concentrations
]
simplices = [
    GammaSimplex(Dirichlet(concentration))
    for concentration in concentrations
]

r2d2s = [
    R2D2(0, 0, rand(rng, simplices), ScaleHierarchy([], rand(rng, n_parameters)))
    for concentration in concentrations
]

n_functions = 3
hsgps = [
    HSGP(
        MeanShift(Normal(), randn(rng, n_functions)), 
        Normal(), 
        Normal(), 
        # ScaleHierarchy([], 1e-3.+zeros(n_functions))
        ScaleHierarchy([], rand(rng, n_functions))
    )
    for concentration in concentrations
]

WarmupHMC.reparametrization_parameters(::Any) = Float64[]
@testset "All Tests" begin
    gammas = Gamma.(exp.(randn(rng, 100)), exp.(randn(rng, 100)))
    qs = rand(rng, 100)
    @testset "Sensitivities" begin
        # @testset "Gamma" sensitivity_tests(gammas, qs)
    end
    @testset "Transformation tests" begin
        # @testset "ScaleHierarchy" transformation_tests(hierarchies)
        # @testset "MeanShift" transformation_tests(mean_shifts)
        # @testset "GammaSimplex" transformation_tests(simplices)
        # @testset "R2D2" transformation_tests(r2d2s)
        @testset "HSGP" transformation_tests(hsgps, (test_draws(source)); iterations=50)
    end
end

loss_matrix(parametrizations) = begin
    draws = (test_draws(parametrizations[1]))
    rv = pairwise(parametrizations) do lhs, rhs
        parameters = WarmupHMC.reparametrization_parameters(rhs)
        loss = WarmupHMC.reparametrization_loss_function(lhs, draws)
        loss(parameters)
    end
    rv .-= diag(rv)
    rv
end
diff_matrix(parametrizations) = pairwise(parametrizations) do lhs, rhs
    rmse(WarmupHMC.reparametrization_parameters.((lhs, rhs))...)
end
scatter_matrix(parametrizations) = begin 
    draws = test_draws(parametrizations[1], n_draws=1000)
    pairwise(parametrizations) do lhs, rhs
        rdraws = WarmupHMC.reparametrize(lhs, rhs, draws)
        scatter(rdraws[1,:], rdraws[2,:])
    end
end
plot(scatter_matrix(mean_shifts[1:4])..., size=(1600, 1600))
loss_matrix(mean_shifts)