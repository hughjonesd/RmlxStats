skip_if_not_installed("glmnet")

test_that("mlxs_glmnet matches glmnet for gaussian lasso", {
  set.seed(42)
  n <- 100
  p <- 20
  x <- matrix(rnorm(n * p), nrow = n, ncol = p)
  beta_true <- c(runif(5, -1, 1), rep(0, p - 5))
  y <- drop(x %*% beta_true + rnorm(n))
  lambda <- 0.1

  ref <- glmnet::glmnet(x, y, family = "gaussian", alpha = 1, lambda = lambda, standardize = TRUE)
  fit <- mlxs_glmnet(x, y, family = mlxs_gaussian(), alpha = 1, lambda = lambda,
                     standardize = TRUE, maxit = 3000, tol = 1e-6)

  expect_equal(as.matrix(fit$beta)[, 1], as.numeric(ref$beta), tolerance = 1e-2)
  expect_equal(fit$a0[1], as.numeric(ref$a0), tolerance = 1e-2)
})

test_that("mlxs_glmnet matches glmnet for binomial lasso", {
  set.seed(99)
  n <- 150
  p <- 15
  x <- matrix(rnorm(n * p), nrow = n, ncol = p)
  coef_true <- c(runif(4, -1, 1), rep(0, p - 4))
  linpred <- drop(x %*% coef_true)
  prob <- 1 / (1 + exp(-linpred))
  y <- rbinom(n, size = 1, prob = prob)
  lambda <- 0.05

  ref <- glmnet::glmnet(x, y, family = "binomial", alpha = 1, lambda = lambda, standardize = TRUE)
  fit <- mlxs_glmnet(x, y, family = mlxs_binomial(), alpha = 1, lambda = lambda,
                     standardize = TRUE, maxit = 4000, tol = 1e-6)

  expect_equal(as.matrix(fit$beta)[, 1], as.numeric(ref$beta), tolerance = 5e-2)
  expect_equal(fit$a0[1], as.numeric(ref$a0), tolerance = 5e-2)
})
