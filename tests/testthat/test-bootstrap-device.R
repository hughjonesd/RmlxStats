test_that("case bootstrap works without a GPU backend", {
  Rmlx::local_device("cpu")
  fit <- mlxs_lm(mpg ~ cyl + disp, data = mtcars)
  set_device <- Rmlx::mlx_device
  testthat::local_mocked_bindings(
    mlx_has_gpu = function() FALSE,
    mlx_device = function(value) {
      if (missing(value)) return(set_device())
      if (value == "gpu") stop("No GPU backend")
      set_device(value)
    },
    .package = "Rmlx"
  )

  boot <- summary(fit, bootstrap = TRUE,
    bootstrap_args = list(B = 10, seed = 123, progress = FALSE)
  )
  ci <- confint(fit, bootstrap = TRUE,
    bootstrap_args = list(B = 10, seed = 123, progress = FALSE)
  )

  expect_true(all(is.finite(as.numeric(boot$bootstrap$se))))
  expect_equal(dim(ci), c(3L, 2L))
  expect_true(all(is.finite(ci)))
  expect_identical(Rmlx::mlx_device(), "cpu")
})
