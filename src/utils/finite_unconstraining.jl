
finite_logit(x, reg=1e-4) = logit.(.5 .+ (x .- .5) .* (1 .- reg))
finite_log(x, reg=1e-16) = log.(x .+ reg)