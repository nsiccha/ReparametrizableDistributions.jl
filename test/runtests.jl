using TestEnv; TestEnv.activate("ReparametrizableDistributions");
using WarmupHMC, ReparametrizableDistributions, ReverseDiff, Distributions, Random, Test, Optim

rmse(x,y; m=mean) = sqrt(m((x.-y).^2))
pairwise(f, arg, args...) = [
    f(lhs, rhs, args...) for lhs in arg, rhs in arg
]
transformation_tests(parametrizations, draws) = begin 
    # @testset "identity" pairwise(reparametrization_test, parametrizations)
    # @testset "rmse" pairwise(rmse_test, parametrizations, draws)
    # @testset "loss" pairwise(loss_test, parametrizations, draws)
    @testset "easy_convergence" pairwise(easy_convergence_test, parametrizations, draws)
    # @testset "hard_convergence" pairwise(hard_convergence_test, parametrizations, draws)
end 
reparametrization_test(lhs, rhs) = begin 
    parameters = WarmupHMC.reparametrization_parameters.([lhs, rhs])
    rlhs = WarmupHMC.reparametrize(lhs, parameters[2])
    lrlhs = WarmupHMC.reparametrize(rlhs, parameters[1])
    @test lhs == lrlhs
end
rmse_test(lhs, rhs, draws) = begin
    rdraws = WarmupHMC.reparametrize(lhs, rhs, draws)
    lrdraws = WarmupHMC.reparametrize(rhs, lhs, rdraws)
    @test rmse(draws, lrdraws) < 1e-8
end
loss_test(lhs, rhs, draws) = begin
    parameters = WarmupHMC.reparametrization_parameters.([lhs, rhs])
    loss = WarmupHMC.reparametrization_loss_function(lhs, draws)
    lloss, rloss = loss.(parameters)
    @test lloss <= rloss
end
easy_convergence_test(lhs, rhs, draws) = begin 
    flhs = WarmupHMC.find_reparametrization(:ReverseDiff, lhs, draws)
    lp, rp, fp = WarmupHMC.reparametrization_parameters.([lhs, rhs, flhs])
    @test rmse(lp, fp) <= rmse(rp, fp)
end
hard_convergence_test(lhs, rhs, draws) = begin 
    rdraws = WarmupHMC.reparametrize(lhs, rhs, draws)
    flhs = WarmupHMC.find_reparametrization(:ReverseDiff, rhs, rdraws)
    lp, rp, fp = WarmupHMC.reparametrization_parameters.([lhs, rhs, flhs])
    @test rmse(lp, fp) <= rmse(rp, fp)
end

rng = Xoshiro(0)
n_parameters = 4
n_draws = 100
xi = randn(rng, (n_parameters, n_draws))
cs = [1, 10]#, 100]
scales = [0]#, 1, 2]

hierarchies = [
    ScaleHierarchy(Normal(), rand(rng, n_parameters-1))
    for i in 1:length(cs)*length(scales)
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

using ChainRulesCore
using Distributions
import ReparametrizableDistributions: _logcdf, _invlogcdf

function ChainRulesCore.rrule(::typeof(_logcdf), d, x::Real)
    lq = _logcdf(d, x)
    function _logcdf_pullback(a)
        ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), a .* exp.(logpdf.(d, x) - lq)
    end
    lq, _logcdf_pullback
end

function ChainRulesCore.rrule(::typeof(_invlogcdf), d, lq::Real)
    x = _invlogcdf(d, lq)
    function _invlogcdf_pullback(a)
        ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), a .* exp.(lq - logpdf.(d, x))
    end
    x, _invlogcdf_pullback
end

import ReverseDiff: TrackedReal
# _invlogcdf(d::Gamma{TrackedReal{Float64, Float64, Nothing}}, x::Real) = begin
#     println(("Caught", d, x)) 
#     println(ReverseDiff.value(d))
#     invlogcdf(d, x)
# end
ReverseDiff.value(d::Gamma{TrackedReal{Float64, Float64, Nothing}}) = begin 
    println("Unwrapping $d ($(eltype(d)))")
    Gamma(ReverseDiff.value.(params(d))...)
end
ReverseDiff.@grad_from_chainrules _logcdf(d, x::TrackedReal)
ReverseDiff.@grad_from_chainrules _invlogcdf(d, x::TrackedReal)
# ReverseDiff.@grad_from_chainrules _invlogcdf(d::Gamma{TrackedReal{Float64, Float64, Nothing}}, x::Real)

ReverseDiff.@grad_from_chainrules _invlogcdf(d::TrackedDistribution, x::Real)
# ReverseDiff.@grad_from_chainrules _invlogcdf(T::Type, P::Tuple{TrackedReal, TrackedReal}, x::Real)


# ReverseDiff.@grad_from_chainrules _invlogcdf(d::UnivariateDistribution, lq::TrackedReal)
# _invlogcdf(d::Gamma, lq::Real) = _gammainvlogcdf(params(d)..., lq)
# _gammainvlogcdf(shape, scale, lq) = invlogcdf(Gamma(shape, scale), lq)
# ReverseDiff.@grad_from_chainrules _gammainvlogcdf(shape::Real, scale::Real, lq::TrackedReal)
# ReverseDiff.@grad_from_chainrules _gammainvlogcdf(shape::TrackedReal, scale::TrackedReal, lq::Real)

# function ChainRulesCore.rrule(::typeof(_gammainvlogcdf), shape::Real, scale::Real, lq::Real)
#     d = Gamma(shape, scale)
#     x = invlogcdf(d, lq)
#     function _gammainvlogcdf_pullback(a)
#         ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), a .* exp.(lq - logpdf.(d, x))
#     end
#     x, _gammainvlogcdf_pullback
# end

# function ChainRulesCore.rrule(::typeof(gammainvlogcdf), shape::Real, scale::Real, lq::Real)
#     d = Gamma(shape, scale)
#     x = invlogcdf(d, lq)
#     function gammainvlogcdf_pullback(a)
#         ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), ChainRulesCore.NoTangent(), a .* exp.(lq - logpdf.(d, x))
#     end
#     x, gammainvlogcdf_pullback
# end
# ReverseDiff.@grad_from_chainrules gammainvlogcdf(shape::TrackedReal, scale::TrackedReal, lq::Real)
# ReverseDiff.@grad_from_chainrules _invlogcdf(d::Gamma{Real}, lq::TrackedReal)
# ReverseDiff.@grad_from_chainrules _invlogcdf(d::Gamma{TrackedReal}, lq::Real)
# ReverseDiff.@grad_from_chainrules _invlogcdf(d::Gamma{TrackedReal}, lq::TrackedReal)

# import Distributions: gammainvlogcdf, gammalogcdf
# gammainvlogcdf(k::ReverseDiff.TrackedReal, theta::ReverseDiff.TrackedReal, lq::ReverseDiff.TrackedReal) = ReverseDiff.track(gammainvlogcdf, k, theta, lq)
# ReverseDiff.@grad function gammainvlogcdf(tk, ttheta, tlq)
#     k, theta, lq = ReverseDiff.value.([tk, ttheta, tlq]) 
#     x = gammainvlogcdf(k, theta, lq)
#     return x, a -> (nothing, nothing, nothing)
# end
# gammalogcdf(k::ReverseDiff.TrackedReal, theta::ReverseDiff.TrackedReal, x::ReverseDiff.TrackedReal) = ReverseDiff.track(gammalogcdf, k, theta, x)
# ReverseDiff.@grad function gammalogcdf(tk, ttheta, tx)
#     k, theta, x = ReverseDiff.value.([tk, ttheta, tx]) 
#     lq = gammalogcdf(k, theta, x)
#     return lq, a -> (nothing, nothing, nothing)
# end

@testset "Transformation tests" begin
    # @testset "ScaleHierarchy" transformation_tests(hierarchies, xi)
    @testset "GammaSimplex" transformation_tests(simplices, xi)
end