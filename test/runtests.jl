using TestEnv; TestEnv.activate("ReparametrizableDistributions");
using WarmupHMC, ReparametrizableDistributions, ReverseDiff, Distributions, Random, Test, Optim, ChainRulesTestUtils, NaNStatistics, Plots, LinearAlgebra, DynamicPlots
using DynamicHMC, LogDensityProblemsAD

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
test_draws(lhs; seed=0, rng=Xoshiro(seed), n_draws=100) = randn(rng, (length(lhs), n_draws))
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
    GammaSimplex(Dirichlet(ones(length(concentration))), Dirichlet(concentration))
    for concentration in concentrations
]

r2d2s = [
    R2D2(Normal(), Normal(), rand(rng, simplices), ScaleHierarchy([], rand(rng, n_parameters)))
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

# Directional = ReparametrizableDistributions.Directional

test_draws(lhs::Directional; seed=0, rng=Xoshiro(seed), n_draws=100) = begin
    # return randn(rng, (length(lhs), n_draws))
    location = randn(rng, length(lhs))
    # location .*= (median(lhs.info.radius)) / norm(location)
    location .*= sqrt(median(lhs.info.radius_squared)) / norm(location)
    location .+ randn(rng, (length(lhs), n_draws))
end
directionals2 = [
    Directional(2, exp(4 + 4 * randn(rng)))
    for _ in 1:8
]
# directionals2 = Directional.(2, exp.(3 .+ 2 .* randn(rng, 8)))
# directionals4 = Directional.(4, exp.(randn(rng, length(concentrations))))

# plots = broadcast(directionals2, reshape(directionals2, (1,:))) do lhs, rhs 
#     println("====================")
#     d0 = Directional(2, 0)
#     @time draws = test_draws(lhs, rng=rng)
#     @time parameters = WarmupHMC.reparametrization_parameters.([lhs, rhs])
#     @time loss = WarmupHMC.reparametrization_loss_function(lhs, draws)
#     @time lloss, rloss = loss.(parameters)
#     @time rdraws = WarmupHMC.reparametrize(lhs, rhs, draws)
#     @time zdraws = WarmupHMC.reparametrize(lhs, d0, draws)
#     # rdraws = test_draws(rhs, rng=rng)
#     @time Scatter(eachrow(draws)..., label=lloss) + Scatter(eachrow(rdraws)..., label=rloss) + Scatter(eachrow(zdraws)..., label=loss([-Inf])) + Scatter([0], [0], color=:black)
#     # println("====================")
#     # scatter!(p, eachrow(tdraws)...)
# end
# Figure(plots)

# WarmupHMC.reparametrize(source::ADGradientWrapper, )

# using WarmupHMC, Optim, ReverseDiff
# WarmupHMC.find_reparametrization(::Val{:ReverseDiff}, source, draws::AbstractMatrix; iterations=5, method=LBFGS(), verbose=false) = begin 
#     loss = WarmupHMC.reparametrization_loss_function(source, draws)
#     init_arg = WarmupHMC.reparametrization_parameters(source)
#     # loss_tape = ReverseDiff.compile(ReverseDiff.GradientTape(loss, init_arg))
#     loss_g!(g, arg) = ReverseDiff.gradient!(g, loss, arg)
#     optimization_result = optimize(
#         loss, loss_g!, init_arg, method, 
#         Optim.Options(iterations=iterations)
#     )
#     verbose && display(optimization_result)
#     WarmupHMC.reparametrize(source, Optim.minimizer(optimization_result))
# end

@testset "All Tests" begin
    n_sensitivity_tests = 10
    gammas = Gamma.(exp.(randn(rng, n_sensitivity_tests)), exp.(randn(rng, n_sensitivity_tests)))
    ncss = NoncentralChisq.(exp.(randn(rng, n_sensitivity_tests)), exp.(randn(rng, n_sensitivity_tests)))
    qs = rand(rng, n_sensitivity_tests)
    @testset "Sensitivities" begin
        # @testset "Gamma" sensitivity_tests(gammas, qs)
        @testset "NoncentralChisq" sensitivity_tests(ncss, qs)
    end
    @testset "Transformation tests" begin
        # @testset "ScaleHierarchy" transformation_tests(hierarchies)
        # @testset "MeanShift" transformation_tests(mean_shifts)
        # @testset "GammaSimplex" transformation_tests(simplices)
        # @testset "R2D2" transformation_tests(r2d2s)
        # @testset "HSGP" transformation_tests(hsgps, (test_draws(hsgps[1])); iterations=50)
        # @testset "Directionals2" transformation_tests(directionals2)
        # @testset "Directionals4" transformation_tests(directionals4)
    end
    @testset "DynamicHMC" begin 
        # mcmc_with_warmup(rng, ADgradient(:ReverseDiff, directionals2[1]), 1000)
    end
end


mcmc_with_reparametrization(rng, ADgradient(:ReverseDiff, directionals2[6]), 1000).final_reparametrization_state.reparametrization |> parent |> WarmupHMC.reparametrization_parameters

WarmupHMC.mcmc_keep_reparametrization(
    rng, ADgradient(:ReverseDiff, directionals2[6]), 1000
)

mcmc_with_reparametrization(rng, ADgradient(:ReverseDiff, hierarchies[1]), 1000).final_reparametrization_state.reparametrization |> parent |> WarmupHMC.reparametrization_parameters

mcmc_with_reparametrization(rng, ADgradient(:ReverseDiff, simplices[1]), 1000).final_reparametrization_state.reparametrization |> parent |> WarmupHMC.reparametrization_parameters

mcmc_with_reparametrization(rng, ADgradient(:ReverseDiff, hsgps[1]), 1000)

WarmupHMC.reparametrization_parameters(::Any) = Float64[]
mcmc_with_reparametrization(rng, ADgradient(:ReverseDiff, r2d2s[1]), 1000)

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
# plot(scatter_matrix(mean_shifts[1:4])..., size=(1600, 1600))
# loss_matrix(mean_shifts)

LocScaleHierarchy = ReparametrizableDistributions.LocScaleHierarchy

WarmupHMC.lpdf_and_invariants(source::UnivariateDistribution, draw::AbstractVector, lpdf=0.) = begin 
    lpdf += sum(logpdf.(source, draw))
    (;lpdf, draw)
end

WarmupHMC.lja_reparametrize(source::UnivariateDistribution, target::UnivariateDistribution, invariants::NamedTuple, lja=0.) = begin 
    lja, invariants.draw
end

using Logging, LogDensityProblems
debug_logger = ConsoleLogger(stderr, Logging.Debug)
    
with_logger(debug_logger) do
    n = 16
    post = heteroskedastic_gp(
        rand(Uniform(-1,1), 10), missing,
        1.5, randn(n), zeros(n), randn(n), zeros(n)
    )
    xi = randn(length(post))
    xi |> display
    WarmupHMC.lpdf_and_invariants(post, xi) |> display
    WarmupHMC.lja_reparametrize(post, post, xi) |> display
    rv = mcmc_with_reparametrization(rng, ADgradient(:ReverseDiff, post), 1000)
    rv |> display
    WarmupHMC.lpdf_and_invariants(post, rv.posterior_matrix[:, 1]) |> display
end

