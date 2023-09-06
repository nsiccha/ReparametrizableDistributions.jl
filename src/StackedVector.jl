struct StackedVector{B,V} <: AbstractVector{eltype(V)}
    boundaries::B
    data::V
end
struct Length{C} size::C end
total_length(what) = sum(length.(values(what)))
total_length(what::Length) = sum(values(what))

Base.eltype(::Type{StackedVector{B,V}}) where {B,V} = eltype(V)
# Base.eltype(what::StackedVector) = eltype(what.data)
Base.size(what::StackedVector) = size(what.data)
Base.IndexStyle(::Type{StackedVector{B,V}}) where {B,V} = IndexStyle(V)
Base.getindex(what::StackedVector, i::Int) = getindex(what.data, i)
Base.setindex!(what::StackedVector, v, i::Int) = setindex!(what.data, v, i)
Base.iterate(what::StackedVector) = Base.iterate(what.data)
Base.similar(what::StackedVector) = StackedVector(what.boundaries, similar(what.data))
Base.similar(what::StackedVector, type::Type{S}) where {S} = StackedVector(what.boundaries, similar(what.data, type))
Base.copy(what::StackedVector) = StackedVector(what.boundaries, copy(what.data))
Base.oftype(x::StackedVector, y::AbstractVector) = StackedVector(x.boundaries, oftype(x.data, y))

TupleNamedTuple(::Any, values) = tuple(values...)
TupleNamedTuple(proto::NamedTuple, values) = (;zip(keys(proto), values)...)

stack_vector(proto::Length, data::AbstractVector) = StackedVector(
    TupleNamedTuple(proto.size, cumsum(values(proto.size))),
    data
)
stack_vector(proto, data::AbstractVector) = stack_vector(Length(map(length, proto)), data)
stack_vector(proto::StackedVector, data::AbstractVector) = oftype(proto, data)
#     TupleNamedTuple(proto, cumsum(length.(values(proto)))),
#     data
# )
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
# views_sized(proto, data::AbstractVector) = views(stack_vector_sized(proto, data))
views(proto, data::AbstractVector) = views(stack_vector(proto, data))
# Base.getproperty(what::StackedVector, key::Symbol) = hasfield(StackedVector, key) ? getfield(what, key) : getproperty(views(what), key)
Base.map(f::Function, first::StackedVector, args...; kwargs...) = StackedVector(map(f, views(first), args...; kwargs...))

sum_logpdf(dists, xs) = sum(logpdf.(dists, xs))