library(pacman)
p_load(cmdstanr, posterior, JuliaCall)

julia_library("Pkg")
julia_command("Pkg.activate(\"page\")")
julia_library("ReparametrizableDistributions")
julia_library("WarmupHMC")
julia_library("Optim")
julia_library("ReverseDiff")

mod_grp <- cmdstan_model("page/examples/R/dynamic.stan")

ng <- 100
n_obs = 100

data_grp <- list(
  N = n_obs * ng,
  K = ng,
  x = rep(1:ng, n_obs)
)

data_grp$y <- rnorm(ng*n_obs,mean=data_grp$x)


data_grp$centeredness <- rep(0, ng)
fit <- mod_grp$sample(data = data_grp, chains = 4)


d <- fit$draws(c("mu", "log_sigma0"))

log_sigma0 <- t(matrix(extract_variable(d, "log_sigma0")))

mu <- t(matrix(subset_draws(d, "mu"), ncol = ng))

data_grp$centeredness <- julia_call("ReparametrizableDistributions.find_centeredness", log_sigma0, mu, rep(0, ng))

afit <- mod_grp$sample(data = data_grp, chains = 4)
data_grp$centeredness

fit$summary()
afit$summary()
