# ReparametrizableDistributions.jl

Implements functionality to reparametrize distributions/posteriors to make them easier to sample from using MCMC methods.

## Installation

Initialize julia project if not done yet, see e.g. [https://pkgdocs.julialang.org/v1/environments/](https://pkgdocs.julialang.org/v1/environments/).

In short, something like:

```{.bash}
# Start julia REPL with project defined in current directory
julia --project=.
```
From that REPL, run:
```{.julia}
 # Enter julia package manager
]
# Add registered external packages
add Optim ReverseDiff Random Distributions 
# Add reparametrization package
add https://github.com/nsiccha/ReparametrizableDistributions.jl  
```

After doing that, you should be able to run one of the examples  at [https://nsiccha.github.io/ReparametrizableDistributions.jl/](https://nsiccha.github.io/ReparametrizableDistributions.jl/).