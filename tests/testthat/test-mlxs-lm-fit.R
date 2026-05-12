test_that("mlxs_lm_fit matches lm.fit for unweighted inputs", {
  rows <- 1:10
  design <- cbind(1, as.matrix(mtcars[rows, c("cyl", "disp")]))
  response <- mtcars$mpg[rows]

  mlx_fit <- mlxs_lm_fit(
    x = Rmlx::as_mlx(design),
    y = Rmlx::mlx_matrix(response, ncol = 1)
  )
  base_fit <- lm.fit(design, response)

  expect_equal(
    drop(as.matrix(mlx_fit$coefficients)),
    base_fit$coefficients,
    tolerance = 1e-6,
    ignore_attr = TRUE
  )
  expect_equal(
    drop(as.matrix(mlx_fit$fitted.values)),
    base_fit$fitted.values,
    tolerance = 1e-6,
    ignore_attr = TRUE
  )
  expect_equal(
    drop(as.matrix(mlx_fit$residuals)),
    base_fit$residuals,
    tolerance = 1e-5,
    ignore_attr = TRUE
  )
  expect_equal(
    drop(as.matrix(mlx_fit$effects)),
    base_fit$effects[seq_len(ncol(design))],
    tolerance = 1e-5,
    ignore_attr = TRUE
  )
  expect_s3_class(mlx_fit$qr, "mlx_qr")
})

test_that("mlxs_lm_fit applies weights identically to lm.wfit", {
  rows <- 1:12
  design <- cbind(1, as.matrix(mtcars[rows, c("cyl", "disp")]))
  response <- mtcars$mpg[rows]
  weights <- seq_along(rows) / length(rows)

  mlx_fit <- mlxs_lm_fit(
    x = Rmlx::as_mlx(design),
    y = Rmlx::mlx_matrix(response, ncol = 1),
    weights = Rmlx::mlx_matrix(weights, ncol = 1)
  )
  base_fit <- lm.wfit(design, response, w = weights)

  expect_equal(
    drop(as.matrix(mlx_fit$coefficients)),
    base_fit$coefficients,
    tolerance = 1e-6,
    ignore_attr = TRUE
  )
  expect_equal(
    drop(as.matrix(mlx_fit$fitted.values)),
    base_fit$fitted.values,
    tolerance = 1e-6,
    ignore_attr = TRUE
  )
  expect_equal(
    drop(as.matrix(mlx_fit$residuals)),
    base_fit$residuals,
    tolerance = 1e-5,
    ignore_attr = TRUE
  )
})

test_that("mlxs_lm_fit can refine float32 solves in float64", {
  set.seed(2)
  n <- 160
  x1 <- rnorm(n)
  x2 <- x1 + rnorm(n, sd = 1e-5)
  x3 <- rnorm(n)
  design <- cbind(1, x1, x2, x3)
  beta <- c(1, 0.75, -0.5, 0.25)
  response <- drop(design %*% beta) + rnorm(n, sd = 0.01)

  base_fit <- lm.fit(design, response, tol = 1e-16)
  f32_fit <- mlxs_lm_fit(
    x = Rmlx::as_mlx(design),
    y = Rmlx::mlx_matrix(response, ncol = 1)
  )
  refined_fit <- mlxs_lm_fit(
    x = Rmlx::as_mlx(design),
    y = Rmlx::mlx_matrix(response, ncol = 1),
    epsilon_f64 = 1e-6
  )

  f32_fitted_error <- max(abs(
    drop(as.matrix(f32_fit$fitted.values)) - base_fit$fitted.values
  ))
  refined_fitted_error <- max(abs(
    drop(as.matrix(refined_fit$fitted.values)) - base_fit$fitted.values
  ))

  expect_true(refined_fit$refined)
  expect_gt(refined_fit$refinement_iterations, 0)
  expect_equal(Rmlx::mlx_dtype(refined_fit$coefficients), "float64")
  expect_equal(Rmlx::mlx_device(refined_fit$coefficients), "cpu")
  expect_lt(refined_fitted_error, f32_fitted_error)
  expect_lte(
    as.numeric(refined_fit$refinement_final_error),
    as.numeric(refined_fit$refinement_initial_error)
  )
})

test_that("mlxs_lm_fit only switches to float64 above epsilon_f64", {
  set.seed(2)
  n <- 80
  x1 <- rnorm(n)
  x2 <- x1 + rnorm(n, sd = 1e-5)
  design <- cbind(1, x1, x2)
  response <- drop(design %*% c(1, 0.75, -0.5)) + rnorm(n, sd = 0.01)

  fit <- mlxs_lm_fit(
    x = Rmlx::as_mlx(design),
    y = Rmlx::mlx_matrix(response, ncol = 1),
    epsilon_f64 = 1
  )

  expect_false(fit$refined)
  expect_equal(fit$refinement_iterations, 0)
  expect_equal(Rmlx::mlx_dtype(fit$coefficients), "float32")
})
