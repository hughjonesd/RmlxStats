align_pca_columns <- function(estimate, reference) {
  if (!ncol(reference)) {
    return(estimate)
  }

  signs <- sign(colSums(estimate * reference))
  signs[signs == 0] <- 1
  sweep(estimate, 2L, signs, `*`)
}

make_known_pca_fixture <- function(n, p, sdev) {
  k <- length(sdev)

  scores_raw <- qr.Q(qr(matrix(rnorm(n * k), nrow = n, ncol = k)))
  scores_raw <- scale(scores_raw, center = TRUE, scale = FALSE)
  scores_basis <- qr.Q(qr(scores_raw))
  rotation <- qr.Q(qr(matrix(rnorm(p * k), nrow = p, ncol = k)))

  singular_values <- sdev * sqrt(n - 1)
  x <- scores_basis %*% diag(singular_values, nrow = k) %*% t(rotation)

  list(
    x = x,
    rotation = rotation,
    scores = scores_basis %*% diag(singular_values, nrow = k),
    sdev = sdev
  )
}

test_that("mlxs_prcomp matches stats::prcomp on tall data", {
  set.seed(1)
  x <- matrix(rnorm(60), nrow = 15, ncol = 4)

  fit <- mlxs_prcomp(x, center = TRUE, scale. = TRUE)
  ref <- stats::prcomp(x, center = TRUE, scale. = TRUE)

  rotation <- as.matrix(fit$rotation)
  rotation <- align_pca_columns(rotation, ref$rotation)
  scores <- as.matrix(fit$x)
  scores <- align_pca_columns(scores, ref$x)

  expect_equal(as.numeric(fit$sdev), ref$sdev, tolerance = 5e-4)
  expect_equal(unname(rotation), unname(ref$rotation), tolerance = 2e-3)
  expect_equal(unname(scores), unname(ref$x), tolerance = 2e-3)
  expect_equal(as.matrix(fit$center), matrix(ref$center, nrow = 1), tolerance = 1e-6)
  expect_equal(as.matrix(fit$scale), matrix(ref$scale, nrow = 1), tolerance = 1e-6)
  expect_identical(fit$method, "exact")
})

test_that("mlxs_prcomp matches stats::prcomp on wide data", {
  set.seed(2)
  x <- matrix(rnorm(24), nrow = 4, ncol = 6)

  fit <- mlxs_prcomp(x, center = TRUE, scale. = FALSE)
  ref <- stats::prcomp(x, center = TRUE, scale. = FALSE)

  nonzero <- ref$sdev > 1e-7
  rotation <- as.matrix(fit$rotation)[, nonzero, drop = FALSE]
  ref_rotation <- ref$rotation[, nonzero, drop = FALSE]
  rotation <- align_pca_columns(rotation, ref_rotation)

  scores <- as.matrix(fit$x)[, nonzero, drop = FALSE]
  ref_scores <- ref$x[, nonzero, drop = FALSE]
  scores <- align_pca_columns(scores, ref_scores)

  expect_equal(as.numeric(fit$sdev), ref$sdev, tolerance = 5e-4)
  expect_equal(unname(rotation), unname(ref_rotation), tolerance = 2e-3)
  expect_equal(unname(scores), unname(ref_scores), tolerance = 2e-3)
  expect_identical(fit$method, "exact")
})

test_that("mlxs_prcomp randomized path approximates leading components", {
  set.seed(3)
  x <- matrix(rnorm(240), nrow = 30, ncol = 8)

  fit <- mlxs_prcomp(
    x,
    center = TRUE,
    scale. = FALSE,
    rank. = 3,
    oversample = 10,
    n_iter = 2,
    seed = 99
  )
  ref <- stats::prcomp(x, center = TRUE, scale. = FALSE, rank. = 3)

  rotation <- as.matrix(fit$rotation)
  rotation <- align_pca_columns(rotation, ref$rotation)
  scores <- as.matrix(fit$x)
  scores <- align_pca_columns(scores, ref$x)

  expect_equal(as.numeric(fit$sdev), ref$sdev[seq_len(fit$rank)], tolerance = 1e-6)
  expect_equal(unname(rotation), unname(ref$rotation), tolerance = 1e-5)
  expect_equal(unname(scores), unname(ref$x), tolerance = 1e-5)
  expect_identical(fit$method, "randomized")
})

test_that("mlxs_prcomp randomized path recovers known large low-rank structure", {
  set.seed(8)
  fixture <- make_known_pca_fixture(
    n = 400,
    p = 120,
    sdev = c(6, 4.5, 3, 2, 1.25, 0.6)
  )

  fit <- mlxs_prcomp(
    fixture$x,
    center = TRUE,
    scale. = FALSE,
    rank. = length(fixture$sdev),
    oversample = 10,
    n_iter = 2,
    seed = 11
  )

  rotation <- align_pca_columns(as.matrix(fit$rotation), fixture$rotation)
  scores <- align_pca_columns(as.matrix(fit$x), fixture$scores)

  expect_equal(as.numeric(fit$sdev), fixture$sdev, tolerance = 1e-6)
  expect_equal(unname(rotation), unname(fixture$rotation), tolerance = 1e-6)
  expect_equal(unname(scores), unname(fixture$scores), tolerance = 1e-5)
})

test_that("mlxs_prcomp honours tol and rank.", {
  set.seed(4)
  x <- matrix(rnorm(50), nrow = 10, ncol = 5)

  fit_rank <- mlxs_prcomp(x, rank. = 2, seed = 7)
  fit_tol <- mlxs_prcomp(x, tol = 0.9)

  expect_equal(fit_rank$rank, 2L)
  expect_true(fit_tol$rank <= min(dim(x)))
})

test_that("predict.mlxs_prcomp matches stats::predict.prcomp", {
  set.seed(5)
  x <- matrix(rnorm(84), nrow = 14, ncol = 6)
  newx <- matrix(rnorm(30), nrow = 5, ncol = 6)

  fit <- mlxs_prcomp(x, center = TRUE, scale. = TRUE, rank. = 3, seed = 10)
  ref <- stats::prcomp(x, center = TRUE, scale. = TRUE, rank. = 3)

  pred <- as.matrix(predict(fit, newx))
  ref_pred <- stats::predict(ref, newx)
  pred <- align_pca_columns(pred, ref_pred)

  expect_equal(unname(pred), unname(ref_pred), tolerance = 2e-1)
})

test_that("mlxs_prcomp supports retx = FALSE", {
  set.seed(6)
  x <- matrix(rnorm(60), nrow = 12, ncol = 5)

  fit <- mlxs_prcomp(x, retx = FALSE, rank. = 2, seed = 3)

  expect_false("x" %in% names(fit))
  expect_error(predict(fit), "no scores are available")
})

test_that("mlxs_prcomp methods reuse prcomp presentation behavior", {
  set.seed(7)
  x <- matrix(rnorm(60), nrow = 12, ncol = 5)
  colnames(x) <- paste0("V", 1:5)
  rownames(x) <- paste0("r", 1:12)
  fit <- mlxs_prcomp(x, rank. = 3, seed = 4)

  expect_equal(nobs(fit), nrow(x))

  sum_obj <- summary(fit)
  expect_s3_class(sum_obj, "summary.prcomp")
  expect_true("importance" %in% names(sum_obj))

  tidy_obj <- tidy(fit)
  expect_equal(
    names(tidy_obj),
    c("component", "std.dev", "proportion", "cumulative")
  )
  expect_equal(nrow(tidy_obj), fit$rank)

  aug_df <- augment(fit, data = as.data.frame(x))
  expect_equal(nrow(aug_df), nrow(x))
  expect_true(all(paste0(".fitted", fit$component_names) %in% names(aug_df)))

  aug_mlx <- augment(fit, output = "mlx")
  expect_s3_class(aug_mlx, "mlx")
  expect_equal(dim(aug_mlx), c(nrow(x), fit$rank))

  named_new <- x[1:4, c(5, 3, 1, 4, 2), drop = FALSE]
  pred_named <- as.matrix(predict(fit, named_new))
  pred_ref <- stats::predict(
    stats::prcomp(x, rank. = 3),
    named_new
  )
  pred_named <- align_pca_columns(pred_named, pred_ref)
  expect_equal(unname(pred_named), unname(pred_ref), tolerance = 2e-1)

  expect_output(print(fit), "Standard deviations")
  expect_silent(graphics::plot(fit))
  expect_silent(stats::biplot(fit))
})

test_that("mlxs_prcomp errors on constant columns when scaling", {
  x <- cbind(1, matrix(rnorm(20), nrow = 5))

  expect_error(
    mlxs_prcomp(x, scale. = TRUE),
    "cannot rescale a constant/zero column"
  )
})

test_that("mlxs_prcomp validates inputs", {
  x <- matrix(rnorm(20), nrow = 5)

  expect_error(mlxs_prcomp(matrix(c(1, NA, 2, 3), nrow = 2)), "finite")
  expect_error(mlxs_prcomp(x, rank. = 0), "rank.")
  expect_error(mlxs_prcomp(x, tol = -1), "tol")
  expect_error(mlxs_prcomp(x, oversample = -1), "oversample")
  expect_error(mlxs_prcomp(x, n_iter = -1), "n_iter")
  expect_error(
    predict(mlxs_prcomp(x), matrix(rnorm(9), nrow = 3)),
    "correct number of columns"
  )
})
