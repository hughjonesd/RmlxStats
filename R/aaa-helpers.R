# Suppress R CMD check notes for closure variables
utils::globalVariables("compiled")


.mlxs_check_qr_full_rank <- function(qr_fit, x, context) {
  dims <- Rmlx::mlx_shape(x)
  if (dims[2] > dims[1]) {
    stop(
      context,
      " requires a full-rank model matrix; rank-deficient fits are not ",
      "supported.",
      call. = FALSE
    )
  }

  r_diag <- abs(Rmlx::diag(qr_fit$R))
  rank_tol_scale <- max(1e-7, .mlxs_tail_epsilon(qr_fit$R))
  rank_tol <- rank_tol_scale * sqrt(Rmlx::colSums(x * x))
  if (any(r_diag <= rank_tol)) {
    stop(
      context,
      " requires a full-rank model matrix; rank-deficient fits are not ",
      "supported.",
      call. = FALSE
    )
  }
  invisible(TRUE)
}

#' Build a coefficient covariance matrix from a QR decomposition
#'
#' Used by `vcov()` methods after fitting. The helper reuses the stored QR
#' factorization from the linear or final weighted-least-squares solve and keeps
#' the resulting covariance matrix as an MLX array.
#'
#' @param qr_fit QR decomposition object with an upper-triangular `R` member.
#' @param n_coef Number of coefficients in the fitted model.
#' @param scale Scalar multiplier, usually the residual dispersion.
#' @return MLX matrix containing `scale * solve(crossprod(R))`.
#' @noRd
.mlxs_vcov_from_qr <- function(qr_fit, n_coef, scale = 1) {
  if (is.null(qr_fit)) {
    stop(
      "QR decomposition not available; refit model to expose vcov.",
      call. = FALSE
    )
  }
  if (identical(Rmlx::mlx_dtype(qr_fit$R), "float64")) {
    Rmlx::local_device("cpu")
  }
  eye <- Rmlx::mlx_eye(n_coef)
  r_inv <- Rmlx::mlx_solve_triangular(qr_fit$R, eye, upper = TRUE,
                                      device = "cpu")
  scale * (r_inv %*% t(r_inv))
}

#' Weighted sum of squares on MLX arrays
#'
#' Internal reduction used by model diagnostics to aggregate residual-like
#' arrays without moving the computation to R.
#'
#' @param values Values to square and sum; coerced with [Rmlx::as_mlx()].
#' @param weights Optional observation weights; coerced with [Rmlx::as_mlx()].
#' @return Scalar MLX array containing the weighted or unweighted sum of
#'   squares.
#' @noRd
.mlxs_weighted_sum_of_squares <- function(values, weights = NULL) {
  vals <- Rmlx::as_mlx(values)
  if (is.null(weights)) {
    return(Rmlx::mlx_sum(vals * vals))
  }
  w <- Rmlx::as_mlx(weights)
  Rmlx::mlx_sum(w * vals * vals)
}

`%||%` <- function (x, y) {
  if (is.null(x)) y else x
}

.mlxs_tail_epsilon <- function(x) {
  if (inherits(x, "mlx")) {
    if (identical(Rmlx::mlx_dtype(x), "float64")) {
      return(.Machine$double.eps)
    }
    return(1e-6)
  }
  .Machine$double.eps
}

.mlxs_napredict <- function(na_action, value) {
  if (is.null(na_action) || !inherits(na_action, "exclude")) {
    return(value)
  }

  # stats::napredict.exclude() indexes through NA positions; [.mlx rejects
  # those indices, so pad MLX outputs with scatter assignment instead.
  if (identical(Rmlx::mlx_dtype(value), "float64")) {
    Rmlx::local_device("cpu")
  }
  full_n <- nrow(value) + length(na_action)
  keep <- setdiff(seq_len(full_n), as.integer(na_action))
  padded <- Rmlx::mlx_matrix(
    rep(NaN, full_n * ncol(value)),
    nrow = full_n,
    ncol = ncol(value),
    dtype = Rmlx::mlx_dtype(value)
  )
  padded[keep, ] <- value
  padded
}

.mlxs_naresid <- .mlxs_napredict
