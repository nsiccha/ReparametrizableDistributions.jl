# title: LKJCholeskyTransform
# description: Cholesky-factor reparametrization with an LKJ(η) prior. Consumes n + n*(n-1)/2 unconstrained parameters and writes the constrained lower-triangular factor in place.
# section: transforms
# id: lkj_cholesky
# tags: lkj, cholesky, correlation

LKJCholeskyTransform(3; eta=2.0)
