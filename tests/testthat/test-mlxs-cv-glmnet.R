skip_if_not_installed("glmnet")

test_that("mlxs_cv_glmnet returns a cv-style gaussian fit", {
  set.seed(1001)
  n <- 80
  p <- 10
  x <- matrix(rnorm(n * p), nrow = n, ncol = p)
  beta_true <- c(1, -0.8, 0.5, rep(0, p - 3))
  y <- drop(x %*% beta_true + rnorm(n))
  foldid <- sample(rep(1:5, length.out = n))

  fit <- mlxs_cv_glmnet(
    x, y,
    family = mlxs_gaussian(),
    alpha = 1,
    nlambda = 12,
    foldid = foldid,
    maxit = 400,
    tol = 1e-5,
    keep = TRUE
  )

  expect_s3_class(fit, "mlxs_cv_glmnet")
  expect_equal(length(fit$lambda), 12)
  expect_equal(dim(fit$cvraw), c(5, 12))
  expect_equal(length(fit$cvm), 12)
  expect_equal(length(fit$cvsd), 12)
  expect_true(fit$lambda.1se >= fit$lambda.min)
  expect_equal(dim(fit$fit.preval), c(n, 12))
  expect_equal(fit$glmnet.fit$lambda, fit$lambda)
  expect_equal(sort(unique(fit$foldid)), 1:5)
})

test_that("mlxs_cv_glmnet methods return selected coefficients and predictions", {
  set.seed(1002)
  n <- 60
  p <- 8
  x <- matrix(rnorm(n * p), nrow = n, ncol = p)
  y <- drop(x[, 1] - 0.5 * x[, 2] + rnorm(n))

  fit <- mlxs_cv_glmnet(
    x, y,
    family = mlxs_gaussian(),
    alpha = 1,
    nlambda = 10,
    nfolds = 4,
    maxit = 400,
    tol = 1e-5
  )

  coef_min <- coef(fit, s = "lambda.min")
  coef_1se <- coef(fit, s = "lambda.1se")
  pred_min <- predict(fit, x[1:7, , drop = FALSE], s = "lambda.min")
  pred_num <- predict(fit, x[1:7, , drop = FALSE], s = fit$lambda[3])

  expect_equal(nrow(coef_min), p + 1L)
  expect_equal(ncol(coef_min), 1L)
  expect_equal(nrow(coef_1se), p + 1L)
  expect_equal(dim(pred_min), c(7, 1))
  expect_equal(dim(pred_num), c(7, 1))
})

test_that("mlxs_cv_glmnet returns binomial losses and class predictions", {
  set.seed(1003)
  n <- 90
  p <- 9
  x <- matrix(rnorm(n * p), nrow = n, ncol = p)
  eta <- 1.2 * x[, 1] - 0.7 * x[, 2]
  prob <- 1 / (1 + exp(-eta))
  y <- rbinom(n, size = 1, prob = prob)

  fit <- mlxs_cv_glmnet(
    x, y,
    family = mlxs_binomial(),
    alpha = 1,
    nlambda = 8,
    nfolds = 3,
    type.measure = "class",
    maxit = 500,
    tol = 1e-5
  )

  preds <- predict(fit, x[1:10, , drop = FALSE], s = "lambda.min",
                   type = "class")

  expect_equal(unname(fit$name), "Misclassification Error")
  expect_true(all(fit$cvm >= 0 & fit$cvm <= 1))
  expect_true(all(preds %in% c(0, 1)))
})

test_that("mlxs_cv_glmnet tracks manual fold losses", {
  set.seed(1004)
  n <- 50
  p <- 6
  x <- matrix(rnorm(n * p), nrow = n, ncol = p)
  y <- drop(0.8 * x[, 1] - 0.4 * x[, 2] + rnorm(n))
  foldid <- sample(rep(1:5, length.out = n))

  fit <- mlxs_cv_glmnet(
    x, y,
    family = mlxs_gaussian(),
    alpha = 1,
    nlambda = 6,
    foldid = foldid,
    maxit = 400,
    tol = 1e-5
  )

  manual_cvm <- numeric(length(fit$lambda))
  for (j in seq_along(fit$lambda)) {
    fold_losses <- numeric(5)
    for (k in 1:5) {
      train <- foldid != k
      test <- foldid == k
      fold_fit <- mlxs_glmnet(
        x[train, , drop = FALSE],
        y[train],
        family = mlxs_gaussian(),
        alpha = 1,
        lambda = fit$lambda[j],
        maxit = 400,
        tol = 1e-5
      )
      pred <- predict(fold_fit, x[test, , drop = FALSE], type = "response")
      fold_losses[k] <- mean((as.numeric(pred) - y[test])^2)
    }
    manual_cvm[j] <- mean(fold_losses)
  }

  expect_equal(fit$cvm, manual_cvm, tolerance = 1e-6)
})

test_that("mlxs_cv_glmnet is close to cv.glmnet for gaussian", {
  set.seed(1006)
  n <- 80
  p <- 10
  x <- matrix(rnorm(n * p), nrow = n, ncol = p)
  beta_true <- c(1, -0.8, 0.5, rep(0, p - 3))
  y <- drop(x %*% beta_true + rnorm(n))
  foldid <- sample(rep(1:5, length.out = n))

  # Share the lambda grid so cross-validation differences reflect the fit,
  # not path generation.
  path_ref <- glmnet::glmnet(
    x, y,
    family = "gaussian",
    alpha = 1,
    nlambda = 12,
    standardize = TRUE
  )
  lambda <- path_ref$lambda

  ref <- glmnet::cv.glmnet(
    x, y,
    family = "gaussian",
    alpha = 1,
    lambda = lambda,
    foldid = foldid,
    keep = TRUE,
    standardize = TRUE
  )
  fit <- mlxs_cv_glmnet(
    x, y,
    family = mlxs_gaussian(),
    alpha = 1,
    lambda = lambda,
    foldid = foldid,
    keep = TRUE,
    standardize = TRUE,
    maxit = 800,
    tol = 1e-6
  )

  expect_equal(fit$cvm, ref$cvm, tolerance = 1e-2)
  expect_equal(fit$cvsd, ref$cvsd, tolerance = 1e-3)
  expect_equal(fit$index["min", 1], ref$index["min", 1])
  expect_equal(fit$index["1se", 1], ref$index["1se", 1])
  expect_equal(fit$fit.preval, ref$fit.preval, tolerance = 2.5e-2)
  expect_true(isTRUE(all.equal(
    as.numeric(coef(fit, s = "lambda.min")),
    as.numeric(coef(ref, s = "lambda.min")),
    tolerance = 2e-3,
    scale = 1
  )))
})

test_that("mlxs_cv_glmnet is close to cv.glmnet for binomial", {
  set.seed(1007)
  n <- 100
  p <- 10
  x <- matrix(rnorm(n * p), nrow = n, ncol = p)
  eta <- 1.1 * x[, 1] - 0.7 * x[, 2]
  prob <- 1 / (1 + exp(-eta))
  y <- rbinom(n, size = 1, prob = prob)
  foldid <- sample(rep(1:5, length.out = n))

  path_ref <- glmnet::glmnet(
    x, y,
    family = "binomial",
    alpha = 1,
    nlambda = 10,
    standardize = TRUE
  )
  lambda <- path_ref$lambda

  ref <- glmnet::cv.glmnet(
    x, y,
    family = "binomial",
    alpha = 1,
    lambda = lambda,
    foldid = foldid,
    keep = TRUE,
    type.measure = "deviance",
    standardize = TRUE
  )
  fit <- mlxs_cv_glmnet(
    x, y,
    family = mlxs_binomial(),
    alpha = 1,
    lambda = lambda,
    foldid = foldid,
    keep = TRUE,
    type.measure = "deviance",
    standardize = TRUE,
    maxit = 1200,
    tol = 1e-6
  )

  expect_equal(fit$cvm, ref$cvm, tolerance = 2e-3)
  expect_equal(fit$cvsd, ref$cvsd, tolerance = 2e-3)
  expect_equal(fit$index["min", 1], ref$index["min", 1])
  expect_equal(fit$index["1se", 1], ref$index["1se", 1])
  expect_equal(fit$fit.preval, plogis(ref$fit.preval), tolerance = 5e-3)
  expect_lt(
    max(abs(
      as.numeric(coef(fit, s = "lambda.min")) -
      as.numeric(coef(ref, s = "lambda.min"))
    )),
    2e-3
  )
})

test_that("mlxs_cv_glmnet rejects unsupported arguments", {
  set.seed(1005)
  x <- matrix(rnorm(80), nrow = 20, ncol = 4)
  y <- rnorm(20)

  expect_error(
    mlxs_cv_glmnet(x, y, family = mlxs_gaussian(), weights = rep(1, 20)),
    "weights are not implemented"
  )
  expect_error(
    mlxs_cv_glmnet(x, y, family = mlxs_gaussian(), alignment = "fraction"),
    "alignment = 'lambda' only"
  )
  expect_error(
    mlxs_cv_glmnet(x, y, family = mlxs_gaussian(), grouped = FALSE),
    "grouped = TRUE only"
  )
  expect_error(
    mlxs_cv_glmnet(x, y, family = mlxs_gaussian(), parallel = TRUE),
    "parallel = TRUE is not implemented"
  )
  expect_error(
    mlxs_cv_glmnet(
      x, y, family = mlxs_binomial(), y = rbinom(20, 1, 0.5),
      type.measure = "auc"
    ),
    "not implemented"
  )
})
