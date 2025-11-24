skip_if_not_installed("glmnet")

test_that("mlxs_glmnet matches glmnet for gaussian lasso", {
  set.seed(42)
  n <- 100
  p <- 20
  x <- matrix(rnorm(n * p, mean = 3, sd = 2), nrow = n, ncol = p)
  beta_true <- c(runif(5, -1, 1), rep(0, p - 5))
  y <- drop(x %*% beta_true + rnorm(n))
  lambda <- 0.1

  ref <- glmnet::glmnet(x, y, family = "gaussian", alpha = 1, lambda = lambda, standardize = TRUE)
  fit <- mlxs_glmnet(x, y, family = mlxs_gaussian(), alpha = 1, lambda = lambda,
                     standardize = TRUE, maxit = 3000, tol = 1e-6)

  expect_equal(as.matrix(fit$beta)[, 1], as.numeric(ref$beta), tolerance = 5e-2)
  expect_equal(as.numeric(fit$a0)[1], as.numeric(ref$a0), tolerance = 5e-2)
})

test_that("mlxs_glmnet matches glmnet for binomial lasso", {
  set.seed(99)
  n <- 150
  p <- 15
  x <- matrix(rnorm(n * p, mean = 3, sd = 2), nrow = n, ncol = p)
  coef_true <- c(runif(4, -1, 1), rep(0, p - 4))
  linpred <- drop(x %*% coef_true)
  prob <- 1 / (1 + exp(-linpred))
  y <- rbinom(n, size = 1, prob = prob)
  lambda <- 0.05

  ref <- glmnet::glmnet(x, y, family = "binomial", alpha = 1, lambda = lambda, standardize = TRUE)
  fit <- mlxs_glmnet(x, y, family = mlxs_binomial(), alpha = 1, lambda = lambda,
                     standardize = TRUE, maxit = 4000, tol = 1e-6)

  expect_equal(as.matrix(fit$beta)[, 1], as.numeric(ref$beta), tolerance = 5e-2)
  expect_equal(as.numeric(fit$a0)[1], as.numeric(ref$a0), tolerance = 5e-2)
})

test_that("mlxs_glmnet works with standardize = FALSE", {
  set.seed(123)
  n <- 100
  p <- 10
  x <- matrix(rnorm(n * p), nrow = n, ncol = p)
  beta_true <- c(runif(3, -1, 1), rep(0, p - 3))
  y <- drop(x %*% beta_true + rnorm(n))
  lambda <- 0.2

  ref <- glmnet::glmnet(x, y, family = "gaussian", alpha = 1, lambda = lambda, standardize = FALSE)
  fit <- mlxs_glmnet(x, y, family = mlxs_gaussian(), alpha = 1, lambda = lambda,
                     standardize = FALSE, maxit = 3000, tol = 1e-6)

  expect_equal(as.matrix(fit$beta)[, 1], as.numeric(ref$beta), tolerance = 5e-2)
  expect_equal(as.numeric(fit$a0)[1], as.numeric(ref$a0), tolerance = 5e-2)
})

test_that("strong rules produce identical results to non-screened for gaussian", {
  set.seed(456)
  n <- 100
  p <- 30
  n_nonzero <- 3

  # Generate sparse problem
  x <- matrix(rnorm(n * p), nrow = n, ncol = p)
  beta_true <- numeric(p)
  beta_true[sample(p, n_nonzero)] <- rnorm(n_nonzero, sd = 2)
  y <- drop(x %*% beta_true + rnorm(n))

  fit_with_rules <- mlxs_glmnet(x, y, family = mlxs_gaussian(), alpha = 1,
                                nlambda = 20, use_strong_rules = TRUE,
                                maxit = 200)
  fit_no_rules <- mlxs_glmnet(x, y, family = mlxs_gaussian(), alpha = 1,
                              nlambda = 20, use_strong_rules = FALSE,
                              maxit = 200)

  expect_equal(as.matrix(fit_with_rules$beta), as.matrix(fit_no_rules$beta), tolerance = 1e-5)
  expect_equal(as.numeric(fit_with_rules$a0), as.numeric(fit_no_rules$a0), tolerance = 1e-5)
  expect_equal(as.numeric(fit_with_rules$lambda), as.numeric(fit_no_rules$lambda))
})

test_that("strong rules produce identical results to non-screened for binomial", {
  set.seed(789)
  n <- 120
  p <- 20
  n_nonzero <- 3

  # Generate sparse problem
  x <- matrix(rnorm(n * p), nrow = n, ncol = p)
  beta_true <- numeric(p)
  beta_true[sample(p, n_nonzero)] <- rnorm(n_nonzero, sd = 1.5)
  linpred <- drop(x %*% beta_true)
  prob <- 1 / (1 + exp(-linpred))
  y <- rbinom(n, size = 1, prob = prob)

  fit_with_rules <- mlxs_glmnet(x, y, family = mlxs_binomial(), alpha = 1,
                                nlambda = 15, use_strong_rules = TRUE, 
                                maxit = 200)
  fit_no_rules <- mlxs_glmnet(x, y, family = mlxs_binomial(), alpha = 1,
                              nlambda = 15, use_strong_rules = FALSE,
                              maxit = 200)

  expect_equal(as.matrix(fit_with_rules$beta), as.matrix(fit_no_rules$beta), tolerance = 1e-5)
  expect_equal(as.numeric(fit_with_rules$a0), as.numeric(fit_no_rules$a0), tolerance = 1e-5)
})

test_that("strong rules work with elastic net (alpha < 1)", {
  set.seed(321)
  n <- 100
  p <- 20
  x <- matrix(rnorm(n * p), nrow = n, ncol = p)
  beta_true <- c(rnorm(5, sd = 2), rep(0, p - 5))
  y <- drop(x %*% beta_true + rnorm(n))

  fit_with_rules <- mlxs_glmnet(x, y, family = mlxs_gaussian(), alpha = 0.5,
                                 nlambda = 20, use_strong_rules = TRUE)
  fit_no_rules <- mlxs_glmnet(x, y, family = mlxs_gaussian(), alpha = 0.5,
                               nlambda = 20, use_strong_rules = FALSE)

  expect_equal(as.matrix(fit_with_rules$beta), as.matrix(fit_no_rules$beta), tolerance = 1e-5)
  expect_equal(as.numeric(fit_with_rules$a0), as.numeric(fit_no_rules$a0), tolerance = 1e-5)
})

test_that("strong rules work with very sparse problems", {
  set.seed(654)
  n <- 100
  p <- 20
  n_nonzero <- 2

  # Very sparse problem
  x <- matrix(rnorm(n * p), nrow = n, ncol = p)
  beta_true <- numeric(p)
  beta_true[sample(p, n_nonzero)] <- rnorm(n_nonzero, sd = 3)
  y <- drop(x %*% beta_true + rnorm(n))

  fit_with_rules <- mlxs_glmnet(x, y, family = mlxs_gaussian(), alpha = 1,
                                 nlambda = 25, use_strong_rules = TRUE)
  fit_no_rules <- mlxs_glmnet(x, y, family = mlxs_gaussian(), alpha = 1,
                               nlambda = 25, use_strong_rules = FALSE)

  expect_equal(as.matrix(fit_with_rules$beta), as.matrix(fit_no_rules$beta), tolerance = 1e-5)
  expect_equal(as.numeric(fit_with_rules$a0), as.numeric(fit_no_rules$a0), tolerance = 1e-5)

  # Check that some screening actually happened
  # (most lambdas should have fewer than p non-zero coefficients)
  n_nonzero_per_lambda <- colSums(abs(as.matrix(fit_with_rules$beta)) > 1e-8)
  expect_true(median(n_nonzero_per_lambda) < p)
})

test_that("strong rules work with dense problems", {
  set.seed(987)
  n <- 100
  p <- 20

  # Dense problem - all coefficients non-zero
  x <- matrix(rnorm(n * p), nrow = n, ncol = p)
  beta_true <- rnorm(p, sd = 0.5)
  y <- drop(x %*% beta_true + rnorm(n))

  fit_with_rules <- mlxs_glmnet(x, y, family = mlxs_gaussian(), alpha = 1,
                                 nlambda = 15, use_strong_rules = TRUE)
  fit_no_rules <- mlxs_glmnet(x, y, family = mlxs_gaussian(), alpha = 1,
                               nlambda = 15, use_strong_rules = FALSE)

  expect_equal(as.matrix(fit_with_rules$beta), as.matrix(fit_no_rules$beta), tolerance = 1e-5)
  expect_equal(as.numeric(fit_with_rules$a0), as.numeric(fit_no_rules$a0), tolerance = 1e-5)
})
