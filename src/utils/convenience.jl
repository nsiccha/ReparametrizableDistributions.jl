find_centeredness(log_scales::AbstractMatrix, x::AbstractMatrix, centeredness::AbstractVector) = begin 
    scale_hierarchy = ScaleHierarchy([], centeredness)
    find_reparametrization(scale_hierarchy, [
        (;log_scale, weights)
        for (log_scale, weights) in zip(eachcol(log_scales), eachcol(x))
    ]).centeredness
end