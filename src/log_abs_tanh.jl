"""
    log_abs_tanh(x)

Compute `log(|tanh(x)|)` in a numerically stable way.

Uses the identity `tanh(x) = (1 - exp(-2|x|)) / (1 + exp(-2|x|))`, so
`log|tanh(x)| = log(1 - exp(-2|x|)) - log(1 + exp(-2|x|)) = log1mexp(-2|x|) - log1pexp(-2|x|)`.
"""
log_abs_tanh(x) = begin
    z = -2 * abs(x)
    log1mexp(z) - log1pexp(z)
end

"""
    log_square_tanh(x)

Compute `log(tanh(x)^2) = 2 * log|tanh(x)|` in a numerically stable way.
"""
log_square_tanh(x) = 2 * log_abs_tanh(x)
