module ReparametrizableDistributionsTests
using Test, Random, ReparametrizableDistributions, LinearAlgebra, Distributions, LogExpFunctions, FiniteDifferences, TestModules
include("ReparametrizableDistributionsTests.jl")
end

using TestModules
runtests!(ReparametrizableDistributionsTests)
