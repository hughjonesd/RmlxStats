# Suppress R CMD check notes for closure variables
utils::globalVariables("compiled")

#' Resolve coefficient names for fitted MLXS models
#'
#' Internal helper used by summaries, intervals, and bootstrap code when they
#' need the model-matrix column names attached to coefficient-sized outputs.
#'
#' @param object Fitted `mlxs_lm` or `mlxs_glm` object.
#' @return Character vector of coefficient names.
#' @noRd
.mlxs_coef_names <- function(object) {
  if (!is.null(object$coef_names)) {
    return(object$coef_names)
  }
  mm <- stats::model.matrix(object$terms, object$model)
  colnames(mm)
}

#' Reject rank-deficient model matrices
#'
#' Shared pre-fit guard used by formula interfaces before the design matrix is
#' sent to MLX solvers. MLXS model objects do not currently represent aliased
#' coefficients, so rank deficiency is treated as an error.
#'
#' @param design Numeric model matrix.
#' @param context Name of the caller to include in the error message.
#' @return Invisibly returns `TRUE` when `design` is full rank.
#' @noRd
.mlxs_check_full_rank <- function(design, context) {
  qr_rank <- qr(design)$rank
  n_coef <- ncol(design)
  if (qr_rank < n_coef) {
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
