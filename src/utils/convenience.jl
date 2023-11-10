find_centeredness(log_scales::AbstractMatrix, x::AbstractMatrix, centeredness::AbstractVector) = begin 
    @assert size(log_scales, 2) == size(x, 2)
    @assert size(log_scales, 1) == size(x, 1) || size(log_scales, 1) == 1 
    @assert size(x, 1) == length(centeredness)
    scale_hierarchy = ScaleHierarchy([], centeredness)
    find_reparametrization(scale_hierarchy, [
        (;log_scale, weights)
        for (log_scale, weights) in zip(eachcol(log_scales), eachcol(x))
    ]).centeredness
end