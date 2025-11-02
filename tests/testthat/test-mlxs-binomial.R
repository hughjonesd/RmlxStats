mlx_to_vec <- function(x) {
  as.numeric(as.matrix(x))
}

test_that("mlxs_binomial reproduces logit link helpers", {
  skip_if_not_installed("Rmlx")

  fam_ref <- stats::binomial()
  fam_mlx <- mlxs_binomial()

  eta <- c(-5, -1, 0, 2, 5)
  mu <- fam_ref$linkinv(eta)

  eta_mlx <- Rmlx::as_mlx(matrix(eta, ncol = 1))
  mu_mlx <- fam_mlx$linkinv(eta_mlx)

  expect_equal(mlx_to_vec(mu_mlx), mu, tolerance = 1e-6)
  expect_equal(
    mlx_to_vec(fam_mlx$linkfun(mu_mlx)),
    fam_ref$linkfun(mu),
    tolerance = 1e-6
  )
  expect_equal(
    mlx_to_vec(fam_mlx$mu.eta(eta_mlx)),
    fam_ref$mu.eta(eta),
    tolerance = 1e-6
  )

  y <- c(0, 0.1, 0.5, 0.9, 1)
  wt <- rep(1, length(y))
  y_mlx <- Rmlx::as_mlx(matrix(y, ncol = 1))
  wt_mlx <- Rmlx::as_mlx(matrix(wt, ncol = 1))
  dev_resids_mlx <- mlx_to_vec(fam_mlx$dev.resids(y_mlx, mu = mu_mlx, wt = wt_mlx))
  dev_resids_ref <- fam_ref$dev.resids(y, mu = mu, wt = wt)
  expect_false(any(is.nan(dev_resids_mlx)))
  expect_lt(max(abs(dev_resids_ref - dev_resids_mlx)), 1e-5)
})

test_that("mlxs_binomial handles alternative supported links", {
  skip_if_not_installed("Rmlx")

  supported <- c("log", "cloglog", "cauchit")
  eta <- c(-2, -0.5, 0, 0.5, 2)
  eta_mlx <- Rmlx::as_mlx(matrix(eta, ncol = 1))

  for (lnk in supported) {
    fam_ref <- stats::binomial(link = lnk)
    fam_mlx <- mlxs_binomial(link = lnk)

    mu_ref <- fam_ref$linkinv(eta)
    mu_mlx <- fam_mlx$linkinv(eta_mlx)

    expect_equal(mlx_to_vec(mu_mlx), mu_ref, tolerance = 1e-6)
    expect_equal(
      mlx_to_vec(fam_mlx$linkfun(mu_mlx)),
      fam_ref$linkfun(mu_ref),
      tolerance = 1e-6
    )
    expect_equal(
      mlx_to_vec(fam_mlx$mu.eta(eta_mlx)),
      fam_ref$mu.eta(eta),
      tolerance = 1e-6
    )
  }
})

test_that("mlxs_binomial falls back for unsupported links", {
  expect_warning(
    fam_probit <- mlxs_binomial(link = "probit"),
    "not currently MLX-optimised"
  )
  fam_ref <- stats::binomial(link = "probit")
  expect_identical(fam_probit$link, fam_ref$link)
})
