// Comparison of k groups with common variance and
// hierarchical prior for the mean
data {
  int<lower=0> N; // number of observations
  int<lower=0> K; // number of groups
  array[N] int<lower=1, upper=K> x; // discrete group indicators
  vector[N] y; // real valued observations
  vector[K] centeredness;
}
parameters {
  real<lower=0> sigma0; // prior std constrained to be positive
  real<lower=0> sigma; // common std constrained to be positive
  vector[K] mu_z;
}

transformed parameters {
 vector[K] mu = mu_z .* sigma0 .^ (1 - centeredness);
}

model {
  sigma0 ~ normal(0, 100); // weakly informative prior
  mu_z ~ normal(0, sigma0 .^ (centeredness)); // population prior with unknown parameters

  sigma ~ lognormal(0, .5); // weakly informative prior
  y ~ normal(mu[x], sigma); // observation model / likelihood
}

generated quantities {
  real log_sigma0 = log(sigma0);
}
