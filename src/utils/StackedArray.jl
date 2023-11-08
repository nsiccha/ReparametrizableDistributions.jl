struct StackedArray{T,N,B,V<:AbstractArray{T,N}} <: AbstractArray{T,N}
    boundaries::B
    data::V
end
StackedVector{T,B,V} = StackedArray{T,1,B,V}
StackedMatrix{T,B,V} = StackedArray{T,2,B,V}
struct Length{C} size::C end
total_length(what) = sum(length.(values(what)))
total_length(what::Length) = sum(values(what.size))

Base.eltype(::Type{StackedArray{T}}) where {T} = T
# Base.eltype(what::StackedVector) = eltype(what.data)
Base.size(what::StackedArray) = size(what.data)
Base.IndexStyle(::Type{StackedArray{T,N,B,V}}) where {T,N,B,V} = IndexStyle(V)
Base.getindex(what::StackedArray, i::Int) = getindex(what.data, i)
Base.setindex!(what::StackedArray, v, i::Int) = setindex!(what.data, v, i)
Base.iterate(what::StackedArray) = Base.iterate(what.data)
Base.similar(what::StackedArray) = StackedArray(what.boundaries, similar(what.data))
Base.similar(what::StackedArray, type::Type{S}) where {S} = StackedArray(what.boundaries, similar(what.data, type))
Base.copy(what::StackedArray) = StackedArray(what.boundaries, copy(what.data))
Base.oftype(proto::StackedArray, y::AbstractArray) = StackedArray(proto.boundaries, oftype(proto.data, y))

TupleNamedTuple(::Any, values) = tuple(values...)
TupleNamedTuple(proto::NamedTuple, values) = (;zip(keys(proto), values)...)

stack_array(proto::Length, data::AbstractArray) = StackedArray(
    TupleNamedTuple(proto.size, cumsum(values(proto.size))),
    data
)
stack_array(proto, data::AbstractArray) = stack_array(Length(map(length, proto)), data)
stack_array(proto::StackedArray, data::AbstractArray) = StackedArray(proto.boundaries, data)
#     TupleNamedTuple(proto, cumsum(length.(values(proto)))),
#     data
# )
StackedArray(what) = stack_array(what, vcat(values(what)...))
general_slice(what::StackedArray, f, args...) = TupleNamedTuple(
    what.boundaries,
    (
        f(what.data, range(1, what.boundaries[1]), args...), 
        f.(
            [what.data], 
            range.(1 .+ values(what.boundaries)[1:end-1], values(what.boundaries)[2:end]),
            args...
        )...
    )
)
arrays(what::StackedVector) = general_slice(what, getindex)
arrays(what::StackedMatrix) = general_slice(what, getindex, :)
arrays(proto, data::AbstractArray) = arrays(stack_array(proto, data))
views(what::StackedVector) = general_slice(what, view)
views(what::StackedMatrix) = general_slice(what, view, :)
views(proto, data::AbstractArray) = views(stack_array(proto, data))
# views_sized(proto, data::AbstractArray) = views(stack_vector_sized(proto, data))
# Base.getproperty(what::StackedVector, key::Symbol) = hasfield(StackedVector, key) ? getfield(what, key) : getproperty(views(what), key)
# Base.map(f::Function, first::StackedVector, args...; kwargs...) = StackedVector(map(f, views(first), args...; kwargs...))