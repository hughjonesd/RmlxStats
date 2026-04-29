# Suppress R CMD check notes for closure variables
utils::globalVariables("compiled")

.mlxs_coef_names <- function(object) {
  if (!is.null(object$coef_names)) {
    return(object$coef_names)
  }
  mm <- stats::model.matrix(object$terms, object$model)
  colnames(mm)
}

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

.mlxs_vcov_from_qr <- function(qr_fit, n_coef, scale = 1) {
  if (is.null(qr_fit)) {
    stop(
      "QR decomposition not available; refit model to expose vcov.",
      call. = FALSE
    )
  }
  eye <- Rmlx::mlx_eye(n_coef)
  r_inv <- Rmlx::mlx_solve_triangular(qr_fit$R, eye, upper = TRUE)
  scale * (r_inv %*% t(r_inv))
}

.mlxs_weighted_sum_of_squares <- function(values, weights = NULL) {
  vals <- Rmlx::as_mlx(values)
  if (is.null(weights)) {
    return(Rmlx::mlx_sum(vals * vals))
  }
  w <- Rmlx::as_mlx(weights)
  Rmlx::mlx_sum(w * vals * vals)
}
