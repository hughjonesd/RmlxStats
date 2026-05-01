#' Generate a structured PCA fuzz case.
#'
#' @param seed Integer seed.
#' @param scenario PCA fuzz scenario name.
#' @param n Number of observations.
#' @param p Number of predictors.
#' @param rank_true Rank of the low-rank signal.
#' @param noise_sd Gaussian noise standard deviation.
#'
#' @return A numeric matrix with observations in rows.
#' @noRd
prcomp_fuzz_case <- function(
  seed,
  scenario = c(
    "gaussian_full", "low_rank", "spiked", "clustered_singular",
    "near_duplicate", "large_mean", "sparse_dense"
  ),
  n,
  p,
  rank_true,
  noise_sd = 0
) {
  scenario <- match.arg(scenario)
  set.seed(seed)
  if (scenario == "gaussian_full") {
    x <- matrix(rnorm(n * p), nrow = n)
    colnames(x) <- paste0("x", seq_len(p))
    return(x)
  }
  rank_true <- min(rank_true, n - 1L, p)
  left_raw <- qr.Q(qr(matrix(rnorm(n * rank_true), nrow = n)))
  left_raw <- scale(left_raw, center = TRUE, scale = FALSE)
  left <- qr.Q(qr(left_raw))
  right <- qr.Q(qr(matrix(rnorm(p * rank_true), nrow = p)))
  sdev <- switch(
    scenario,
    low_rank = seq(6, 1.5, length.out = rank_true),
    spiked = c(8, 4, seq(1.5, 0.4, length.out = rank_true - 2L)),
    clustered_singular = rep(3, rank_true) + seq(0, 0.02, length.out = rank_true),
    near_duplicate = seq(5, 1, length.out = rank_true),
    large_mean = seq(5, 1, length.out = rank_true),
    sparse_dense = seq(5, 1, length.out = rank_true)
  )
  x <- left %*% diag(sdev * sqrt(n - 1), nrow = rank_true) %*% t(right)
  x <- x + matrix(rnorm(n * p, sd = noise_sd), nrow = n)

  if (scenario == "near_duplicate" && p >= 4L) {
    x[, p] <- x[, 1L] + rnorm(n, sd = max(noise_sd, 1e-6))
    x[, p - 1L] <- x[, 2L]
  }
  if (scenario == "large_mean") {
    x <- sweep(x, 2L, seq(1e5, 1e6, length.out = p), `+`)
  }
  if (scenario == "sparse_dense") {
    mask <- matrix(runif(n * p) < 0.9, nrow = n)
    x[mask] <- 0
    x <- x + matrix(rnorm(n * p, sd = max(noise_sd, 1e-4)), nrow = n)
  }

  colnames(x) <- paste0("x", seq_len(p))
  x
}

#' Relative distance between PCA loading subspaces.
#'
#' @param estimate Estimated loading matrix.
#' @param reference Reference loading matrix.
#'
#' @return Relative Frobenius distance between projectors.
#' @noRd
prcomp_projector_error <- function(estimate, reference) {
  estimate <- as.matrix(estimate)
  reference <- as.matrix(reference)
  if (!ncol(estimate) && !ncol(reference)) {
    return(0)
  }
  estimate_projector <- estimate %*% t(estimate)
  reference_projector <- reference %*% t(reference)
  sqrt(sum((estimate_projector - reference_projector)^2)) /
    sqrt(max(sum(reference_projector^2), 1e-12))
}

#' Relative PCA reconstruction error.
#'
#' @param x Original numeric data matrix.
#' @param rotation PCA loading matrix.
#' @param center Centering parameter passed to [base::scale()].
#' @param scale Scaling parameter passed to [base::scale()].
#'
#' @return Relative Frobenius reconstruction error for centred/scaled data.
#' @noRd
prcomp_reconstruction_error <- function(x, rotation, center, scale) {
  x_scaled <- scale(x, center = center, scale = scale)
  scores <- x_scaled %*% rotation
  residual <- x_scaled - scores %*% t(rotation)
  sqrt(sum(residual^2)) / sqrt(max(sum(x_scaled^2), 1e-12))
}

#' Summarise one mlxs_prcomp fit against stats::prcomp.
#'
#' @param x Input data matrix.
#' @param fit MLX-backed PCA fit.
#' @param ref Reference [stats::prcomp()] fit.
#' @param scenario Human-readable scenario name.
#' @param case_type Summary row family.
#' @param rank_true True generating rank for synthetic low-rank signal.
#' @param noise_sd Synthetic noise standard deviation.
#' @param center,scale. PCA preprocessing options.
#'
#' @return A one-row data frame of PCA fuzz metrics.
#' @noRd
summarise_prcomp_fit <- function(
  x,
  fit,
  ref,
  scenario,
  case_type,
  rank_true,
  noise_sd,
  center,
  scale.
) {
  rank <- fit$rank
  rotation <- as.matrix(fit$rotation)
  ref_rotation <- ref$rotation[, seq_len(rank), drop = FALSE]
  sdev <- as.numeric(fit$sdev)
  ref_sdev <- ref$sdev[seq_len(rank)]
  # Near-zero singular values have arbitrary bases, so subspace and sdev
  # comparisons use only reference components that are numerically identifiable.
  stable <- ref_sdev > max(ref_sdev) * 1e-6
  if (!any(stable)) {
    stable <- seq_along(ref_sdev) == 1L
  }
  stable_sdev <- sdev[stable]
  stable_ref_sdev <- ref_sdev[stable]
  stable_rotation <- rotation[, stable, drop = FALSE]
  stable_ref_rotation <- ref_rotation[, stable, drop = FALSE]
  prop_var <- stable_sdev^2 / sum(stable_sdev^2)
  ref_prop_var <- stable_ref_sdev^2 / sum(stable_ref_sdev^2)
  reconstruction <- prcomp_reconstruction_error(x, rotation, center, scale.)
  ref_reconstruction <- prcomp_reconstruction_error(
    x,
    ref_rotation,
    center,
    scale.
  )

  fuzz_metric_rows(
    list(
      case_type = case_type,
      scenario = scenario,
      n = nrow(x),
      p = ncol(x),
      rank = rank,
      rank_true = rank_true,
      method = fit$method,
      noise_sd = noise_sd
    ),
    measure     = c("error",     "ratio",     "error",     "loss",           "loss",           "delta",          "error",              "diagnostic",    "diagnostic"),
    target      = c("pca_sdev",  "pca_sdev",  "subspace",  "reconstruction", "reconstruction", "reconstruction", "explained_variance", "orthogonality", "finite"),
    source      = c("mlx",       "mlx",       "mlx",       "mlx",            "reference",      "mlx",            "mlx",                "mlx",           "mlx"),
    baseline    = c("reference", "reference", "reference", NA,               NA,               "reference",      "reference",          "ideal",         NA),
    aggregation = c("max",       "rmse",      "value",     "value",          "value",          "delta",          "max",                "max",           "all"),
    value = c(
      max(abs(stable_sdev - stable_ref_sdev)),
      sqrt(mean((stable_sdev - stable_ref_sdev)^2)) /
        max(sqrt(mean(stable_ref_sdev^2)), 1e-12),
      prcomp_projector_error(stable_rotation, stable_ref_rotation),
      reconstruction,
      ref_reconstruction,
      reconstruction - ref_reconstruction,
      max(abs(prop_var - ref_prop_var)),
      max(abs(crossprod(rotation) - diag(rank))),
      as.numeric(all(is.finite(c(
        sdev,
        rotation,
        reconstruction,
        ref_reconstruction
      ))))
    )
  )
}
