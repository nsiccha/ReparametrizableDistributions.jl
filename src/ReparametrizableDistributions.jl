module ReparametrizableDistributions

export Ignore, ScaleHierarchy, GammaSimplex

using WarmupHMC, Distributions, LogDensityProblems

struct StackedVector{B,V} <: AbstractVector{eltype(V)}
    boundaries::B
    data::V
end
Base.size(what::StackedVector) = size(what.data)
Base.IndexStyle(::Type{StackedVector{B,V}}) where {B,V} = IndexStyle(V)
Base.getindex(what::StackedVector, i::Int) = getindex(what.data, i)
Base.setindex!(what::StackedVector, v, i::Int) = setindex!(what.data, v, i)
Base.iterate(what::StackedVector) = Base.iterate(what.data)
Base.similar(what::StackedVector) = StackedVector(what.boundaries, similar(what.data))
Base.similar(what::StackedVector, type::Type{S}) where {S} = StackedVector(what.boundaries, similar(what.data, type))
Base.oftype(x::StackedVector, y::AbstractVector) = StackedVector(x.boundaries, oftype(x.data, y))

TupleNamedTuple(proto, values) = tuple(values...)
TupleNamedTuple(proto::NamedTuple, values) = (;zip(keys(proto), values)...)

stack_vector(proto, data::AbstractVector) = StackedVector(
    TupleNamedTuple(proto, cumsum(length.(values(proto)))),
    data
)
StackedVector(what) = stack_vector(what, vcat(values(what)...))
general_slice(what::StackedVector, f) = TupleNamedTuple(
    what.boundaries,
    (
        f(what.data, range(1, what.boundaries[1])), 
        f.(
            [what.data], 
            range.(1 .+ values(what.boundaries)[1:end-1], values(what.boundaries)[2:end])
        )...
    )
)
vectors(what::StackedVector) = general_slice(what, getindex)
views(what::StackedVector) = general_slice(what, view)
vectors(proto, data::AbstractVector) = vectors(stack_vector(proto, data))
views(proto, data::AbstractVector) = views(stack_vector(proto, data))

struct Ignore end
Base.:+(lhs::Ignore, rhs::Real) = lhs
Base.:-(lhs::Ignore, rhs::Real) = lhs

abstract type AbstractReparametrizableDistribution end

fixed(source::AbstractReparametrizableDistribution) = source.fixed
variable(source::AbstractReparametrizableDistribution) = source.variable
LogDensityProblems.dimension(source::AbstractReparametrizableDistribution) = length(source)
LogDensityProblems.logdensity(source::AbstractReparametrizableDistribution, draw::AbstractVector) = WarmupHMC.logdensity_and_stuff(source, draw)[1]

struct ScaleHierarchy{F,V} <: AbstractReparametrizableDistribution
    fixed::F
    variable::V
end
WarmupHMC.reparametrization_parameters(source::ScaleHierarchy) = variable(source)
WarmupHMC.reparametrize(source::ScaleHierarchy, parameters::AbstractVector) = ScaleHierarchy(
    fixed(source), parameters
)

centeredness(source::ScaleHierarchy) = variable(source)
Base.length(source::ScaleHierarchy) = 1+length(centeredness(source))

log_scale_distribution(source::ScaleHierarchy, ::Any) = Product([fixed(source)])
hierarchical_distribution(source::ScaleHierarchy, draw::AbstractVector) = begin 
    log_scale = draw[1]
    scales = exp.(log_scale .* centeredness(source))
    Product(Normal.(0, scales))
end
subdistributions(source::ScaleHierarchy, draw::AbstractVector) = log_scale_distribution(source, draw), hierarchical_distribution(source, draw)

@views WarmupHMC.logdensity_and_stuff(source::ScaleHierarchy, draw::AbstractVector, lpdf=0.) = begin 
    sdists = subdistributions(source, draw)
    log_scale, xic = subdraws = views(sdists, draw)
    lpdf += sum(logpdf.(sdists, subdraws))
    x = xic .* exp.(log_scale .* (1 .- centeredness(source)))
    lpdf, x
end

WarmupHMC.lja_reparametrize(source::ScaleHierarchy, target::ScaleHierarchy, draw::AbstractVector, lja=0.) = begin 
    log_scale, sxic = views(subdistributions(source, draw), draw)
    txic = xic .* exp.(log_scale .* (centeredness(target) .- centeredness(source)))
    tdraw = vcat(log_scale, txic)
    lja += logpdf(hierarchical_distribution(target, tdraw), txic)
    lja, tdraw
end

struct GammaSimplex{F,V} <: AbstractReparametrizableDistribution
    fixed::F
    variable::V
end
GammaSimplex(dirichlet::Dirichlet) = GammaSimplex(dirichlet, dirichlet)
WarmupHMC.reparametrization_parameters(source::GammaSimplex) = parametrization_concentrations(source)
WarmupHMC.reparametrize(source::GammaSimplex, parameters::AbstractVector) = GammaSimplex(
    fixed(source), Dirichlet(parameters)
)

target_distribution(source::GammaSimplex) = fixed(source)
Base.length(source::GammaSimplex) = length(target_distribution(source))
parametrization_distribution(source::GammaSimplex) = variable(source)
parametrization_concentrations(source::GammaSimplex) = parametrization_distribution(source).alpha
parametrization_gammas(source::GammaSimplex) = Gamma.(parametrization_concentrations(source))
sum_gamma(source::GammaSimplex) = Gamma(sum(parametrization_concentrations(source)))

_cdf(distribution, x) = cdf(distribution, x)
_quantile(distribution, x) = quantile(distribution, x)
_logcdf(distribution, x) = logcdf(distribution, x)
_invlogcdf(distribution, x) = invlogcdf(distribution, x)

WarmupHMC.logdensity_and_stuff(source::GammaSimplex, draw::AbstractVector, lpdf=0.) = begin 
    lpdf += sum(logpdf.(Normal(), draw)) 
    xi = _invlogcdf.(parametrization_gammas(source), _logcdf.(Normal(), draw))
    x = xi ./ sum(xi)
    lpdf += logpdf(target_distribution(source), x) 
    lpdf -= logpdf(parametrization_distribution(source), x)
    lpdf, x
end

WarmupHMC.lja_reparametrize(source::GammaSimplex, target::GammaSimplex, draw::AbstractVector, lja=0.) = begin 
    sxi = _invlogcdf.(parametrization_gammas(source), _logcdf.(Normal(), draw))
    ssum = sum(sxi)
    tsum = _invlogcdf(sum_gamma(target), _logcdf(sum_gamma(source), ssum))
    txi = sxi .* tsum ./ ssum
    tdraw = _invlogcdf.(Normal(), _logcdf.(parametrization_gammas(target), txi))
    lja += sum(logpdf.(Normal(), tdraw))
    lja -= logpdf(parametrization_distribution(target), txi ./ tsum)
    lja, tdraw
end

end # module ReparametrizableDistributions
