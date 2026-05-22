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

test_that("mlxs_lm_fit accepts full-rank tall design matrices", {
  n <- 10000
  design <- cbind(1, seq_len(n))
  response <- seq_len(n)

  expect_no_error(
    mlxs_lm_fit(
      x = Rmlx::as_mlx(design),
      y = Rmlx::mlx_matrix(response, ncol = 1)
    )
  )
})

test_that("mlxs_lm_fit rejects underdetermined design matrices", {
  design <- matrix(seq_len(12), nrow = 3)
  response <- seq_len(3)

  expect_error(
    mlxs_lm_fit(
      x = Rmlx::as_mlx(design),
      y = Rmlx::mlx_matrix(response, ncol = 1)
    ),
    "full-rank model matrix",
    fixed = TRUE
  )
})

test_that("mlxs_lm_fit rejects rank-deficient design matrices", {
  design <- cbind(1, as.matrix(mtcars[c("cyl", "disp")]))
  design <- cbind(design, disp_copy = design[, "disp"])
  response <- mtcars$mpg

  expect_error(
    mlxs_lm_fit(
      x = Rmlx::as_mlx(design),
      y = Rmlx::mlx_matrix(response, ncol = 1)
    ),
    "full-rank model matrix",
    fixed = TRUE
  )
})
