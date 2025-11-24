
test_that("mlxs_gaussian reproduces gaussian helpers", {
  fam_ref <- stats::gaussian()
  fam_mlx <- mlxs_gaussian()

  eta <- c(-2, -0.5, 0, 0.5, 2)
  mu <- fam_ref$linkinv(eta)
  y <- c(-1, 0.2, 0.5, 1.5, 3)
  wt <- rep(1.5, length(y))

  eta_mlx <- Rmlx::as_mlx(eta)
  mu_mlx <- Rmlx::as_mlx(mu)
  y_mlx <- Rmlx::as_mlx(y)
  wt_mlx <- Rmlx::as_mlx(wt)

  expect_equal(as.numeric(fam_mlx$linkinv(eta_mlx)), mu, tolerance = 1e-6)
  expect_equal(as.numeric(fam_mlx$linkfun(mu_mlx)), fam_ref$linkfun(mu), tolerance = 1e-6)
  expect_equal(as.numeric(fam_mlx$mu.eta(eta_mlx)), fam_ref$mu.eta(eta), tolerance = 1e-6)
  expect_equal(as.numeric(fam_mlx$variance(mu_mlx)), fam_ref$variance(mu), tolerance = 1e-6)
  expect_equal(
    as.numeric(fam_mlx$dev.resids(y_mlx, mu_mlx, wt_mlx)),
    fam_ref$dev.resids(y, mu, wt),
    tolerance = 1e-6
  )
})

test_that("mlxs_poisson reproduces poisson helpers", {
  fam_ref <- stats::poisson()
  fam_mlx <- mlxs_poisson()

  eta <- c(-2, -0.5, 0, 0.5, 2)
  mu <- fam_ref$linkinv(eta)
  y <- c(0, 0, 1, 3, 10)
  wt <- rep(2, length(y))

  eta_mlx <- Rmlx::as_mlx(eta)
  mu_mlx <- Rmlx::as_mlx(mu)
  y_mlx <- Rmlx::as_mlx(y)
  wt_mlx <- Rmlx::as_mlx(wt)

  expect_equal(as.numeric(fam_mlx$linkinv(eta_mlx)), mu, tolerance = 1e-6)
  expect_equal(as.numeric(fam_mlx$linkfun(mu_mlx)), fam_ref$linkfun(mu), tolerance = 1e-6)
  expect_equal(as.numeric(fam_mlx$mu.eta(eta_mlx)), fam_ref$mu.eta(eta), tolerance = 1e-6)
  expect_equal(as.numeric(fam_mlx$variance(mu_mlx)), fam_ref$variance(mu), tolerance = 1e-6)
  expect_equal(
    as.numeric(fam_mlx$dev.resids(y_mlx, mu_mlx, wt_mlx)),
    fam_ref$dev.resids(y, mu, wt),
    tolerance = 1e-5
  )
})

test_that("quasi families wrap correctly", {
  quasi_bin <- mlxs_quasibinomial()
  expect_equal(quasi_bin$family, "quasibinomial")
  expect_true(is.na(quasi_bin$aic(1, 1, 1, 1, 1)))

  quasi_pois <- mlxs_quasipoisson()
  expect_equal(quasi_pois$family, "quasipoisson")
  expect_true(is.na(quasi_pois$aic(1, 1, 1, 1, 1)))
})
