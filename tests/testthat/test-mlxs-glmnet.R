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

  expect_equal(as.numeric(fit$beta), as.numeric(ref$beta), tolerance = 5e-2)
  expect_equal(as.numeric(fit$a0), as.numeric(ref$a0), tolerance = 5e-2)
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

  expect_equal(as.numeric(fit$beta), as.numeric(ref$beta), tolerance = 5e-3)
  expect_equal(as.numeric(fit$a0), as.numeric(ref$a0), tolerance = 5e-2)
})

test_that("mlxs_glmnet null binomial path does not overfit", {
  set.seed(1005)
  n <- 900
  n_test <- 700
  p <- 120
  x <- scale(make_design(n = n, p = p, rho = 0.8))
  x_test <- scale(make_design(n = n_test, p = p, rho = 0.8))
  y <- rbinom(n, size = 1L, prob = plogis(0.1))
  y_test <- rbinom(n_test, size = 1L, prob = plogis(0.1))

  lambda <- glmnet::glmnet(
    x,
    y,
    family = "binomial",
    alpha = 1,
    nlambda = 20,
    lambda.min.ratio = 1e-3,
    standardize = FALSE,
    intercept = TRUE,
    thresh = 1e-12,
    maxit = 100000L
  )$lambda
  ref <- glmnet::glmnet(
    x,
    y,
    family = "binomial",
    alpha = 1,
    lambda = lambda,
    standardize = FALSE,
    intercept = TRUE,
    thresh = 1e-12,
    maxit = 100000L
  )
  fit <- mlxs_glmnet(
    x,
    y,
    family = mlxs_binomial(),
    alpha = 1,
    lambda = lambda,
    standardize = FALSE,
    intercept = TRUE,
    maxit = 5000L,
    tol = 1e-7
  )

  low_lambda <- length(lambda)
  mlx_pred <- predict(fit, newx = x_test, s = lambda[low_lambda],
                      type = "response")
  ref_pred <- predict(ref, newx = x_test, s = lambda[low_lambda],
                      type = "response")
  mlx_loss <- glmnet_fuzz_loss(y_test, mlx_pred, family = "binomial")
  ref_loss <- glmnet_fuzz_loss(y_test, ref_pred, family = "binomial")
  oracle_loss <- glmnet_fuzz_loss(
    y_test,
    rep(plogis(0.1), n_test),
    family = "binomial"
  )

  expect_equal(mlx_loss, ref_loss, tolerance = 1e-3)
  expect_lte(mlx_loss, oracle_loss + 0.1)
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

  expect_equal(as.numeric(fit$beta), as.numeric(ref$beta), tolerance = 1e-6)
  expect_equal(as.numeric(fit$a0), as.numeric(ref$a0), tolerance = 1e-6)
})

test_that("mlxs_glmnet matches glmnet for gaussian without intercept", {
  set.seed(124)
  n <- 120
  p <- 12
  x <- matrix(rnorm(n * p), nrow = n, ncol = p)
  beta_true <- c(runif(4, -1, 1), rep(0, p - 4))
  y <- drop(x %*% beta_true + rnorm(n, sd = 0.5))
  lambda <- 0.15

  ref <- glmnet::glmnet(
    x, y,
    family = "gaussian",
    alpha = 1,
    lambda = lambda,
    standardize = TRUE,
    intercept = FALSE
  )
  fit <- mlxs_glmnet(
    x, y,
    family = mlxs_gaussian(),
    alpha = 1,
    lambda = lambda,
    standardize = TRUE,
    intercept = FALSE,
    maxit = 3000,
    tol = 1e-6
  )

  expect_equal(as.numeric(fit$beta), as.numeric(ref$beta), tolerance = 5e-3)
  expect_equal(as.numeric(fit$a0), as.numeric(ref$a0), tolerance = 1e-6)
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

test_that("mlxs_glmnet stays finite on correlated gaussian designs", {
  set.seed(20251111)
  n <- 1000
  p <- 100
  rho <- 0.3

  x <- matrix(rnorm(n * p), nrow = n, ncol = p)
  for (j in 2:p) {
    x[, j] <- rho * x[, j - 1] + sqrt(1 - rho^2) * x[, j]
  }

  beta_true <- rep(0, p)
  beta_true[seq(10, p, 10)] <- rnorm(length(seq(10, p, 10)), sd = 0.35)
  y <- drop(x %*% beta_true + rnorm(n, sd = 5))

  fit <- mlxs_glmnet(x, y, lambda = 1 / (1:50))
  beta_hat <- as.matrix(coef(fit))

  expect_true(all(is.finite(beta_hat)))
})

test_that("mlxs_glmnet stores paths in MLX and methods can return base or MLX", {
  set.seed(20260514)
  x <- matrix(rnorm(80 * 6), nrow = 80)
  y <- drop(x[, 1] - 0.5 * x[, 2] + rnorm(80))
  lambda <- c(0.2, 0.05)

  fit <- mlxs_glmnet(
    x,
    y,
    family = mlxs_gaussian(),
    lambda = lambda,
    maxit = 80,
    tol = 1e-8,
    tol_f64 = 1e6
  )

  expect_true(inherits(fit$beta, "mlx"))
  expect_true(inherits(fit$a0, "mlx"))
  expect_true(fit$float64)
  expect_equal(fit$float64_reason, "tol_f64")

  coef_base <- coef(fit)
  coef_mlx <- coef(fit, output = "mlx")
  pred_base <- predict(fit, x[1:5, , drop = FALSE])
  pred_mlx <- predict(fit, x[1:5, , drop = FALSE], output = "mlx")

  expect_type(coef_base, "double")
  expect_equal(dim(coef_base), c(ncol(x) + 1L, length(lambda)))
  expect_true(inherits(coef_mlx, "mlx"))
  expect_type(pred_base, "double")
  expect_equal(dim(pred_base), c(5L, length(lambda)))
  expect_true(inherits(pred_mlx, "mlx"))
  expect_equal(as.matrix(pred_mlx), pred_base, tolerance = 1e-10)
})

test_that("mlxs_glmnet switches gaussian Gram path to float64", {
  set.seed(20260515)
  x <- matrix(rnorm(320 * 5), nrow = 320)
  y <- drop(x[, 1] + rnorm(320))

  fit <- mlxs_glmnet(
    x,
    y,
    family = mlxs_gaussian(),
    nlambda = 10,
    maxit = 80,
    tol = 1e-8,
    tol_f64 = 1e6
  )

  expect_true(fit$float64)
  expect_equal(Rmlx::mlx_dtype(fit$beta), "float64")
  expect_output(print(fit), "MLX elastic net fit")
})

test_that("mlxs_glmnet switches binomial path to float64", {
  set.seed(20260516)
  x <- matrix(rnorm(100 * 8), nrow = 100)
  prob <- plogis(0.8 * x[, 1] - 0.5 * x[, 2])
  y <- rbinom(100, size = 1, prob = prob)

  fit <- mlxs_glmnet(
    x,
    y,
    family = mlxs_binomial(),
    nlambda = 4,
    maxit = 80,
    tol = 1e-8,
    tol_f64 = 1e6
  )

  expect_true(fit$float64)
  expect_equal(Rmlx::mlx_dtype(fit$beta), "float64")
  expect_true(all(predict(fit, x[1:6, , drop = FALSE], type = "class") %in%
    c(0, 1)))
})

test_that("mlxs_glmnet handles float64 MLX input immediately on CPU", {
  set.seed(20260517)
  x <- matrix(rnorm(60 * 5), nrow = 60)
  y <- drop(x[, 1] + rnorm(60))

  Rmlx::local_device("cpu")
  x64 <- Rmlx::as_mlx(x, dtype = "float64")

  expect_warning(
    fit <- mlxs_glmnet(
      x64,
      y,
      family = mlxs_gaussian(),
      lambda = c(0.1, 0.02),
      maxit = 50,
      tol = 1e-8,
      tol_f64 = 1e-6
    ),
    "float64 MLX input"
  )

  expect_true(fit$float64)
  expect_equal(fit$float64_reason, "input")
  expect_equal(Rmlx::mlx_dtype(fit$beta), "float64")
})

test_that("mlxs_glmnet validates tol_f64", {
  x <- matrix(rnorm(40), nrow = 10)
  y <- rnorm(10)

  expect_error(
    mlxs_glmnet(x, y, tol = 1e-6, tol_f64 = 1e-6),
    "tol_f64 must be greater than tol"
  )
})
