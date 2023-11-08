
finite_logit(x, reg=1e-4) = logit.(.5 .+ (x .- .5) .* (1 .- reg))
finite_log(x, reg=1e-16) = log.(x .+ reg)

inverse(::typeof(finite_logit)) = logistic
inverse(::typeof(finite_log)) = exp
inverse(::typeof(identity)) = identity