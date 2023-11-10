library(pacman)
p_load(cmdstanr, posterior, JuliaCall)

julia_library("ReparametrizableDistributions")
julia_library("WarmupHMC")
julia_library("Optim")
julia_library("ReverseDiff")


ng <- 4

centeredness <- rep(0, ng)

data_grp <- list(
  N = 10 * ng,
  K = ng,
  x = rep(1:ng, 10),
  centeredness = centeredness
)

data_grp$y <- rnorm(data_grp$x*10)

mod_grp <- cmdstan_model("dynamic.stan")

fit <- mod_grp$sample(data = data_grp, chains = 1)

d <- fit$draws(c("mu", "log_sigma0"))

log_sigma0 <- t(matrix(extract_variable(d, "log_sigma0")))

mu <- t(matrix(subset_draws(d, "mu"), ncol = 5))

centeredness <- julia_call("ReparametrizableDistributions.find_centeredness", log_sigma0, mu, rep(0, ng))