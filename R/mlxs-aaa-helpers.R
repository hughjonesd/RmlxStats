.mlxs_coef_names <- function(object) {
  if (!is.null(object$coef_names)) {
    return(object$coef_names)
  }
  mm <- stats::model.matrix(object$terms, object$model)
  colnames(mm)
}

.mlxs_vcov_from_qr <- function(qr_fit, n_coef, scale = 1) {
  if (is.null(qr_fit)) {
    stop("QR decomposition not available; refit model to expose vcov.", call. = FALSE)
  }
  eye <- Rmlx::mlx_eye(n_coef)
  r_inv <- Rmlx::mlx_solve_triangular(qr_fit$R, eye, upper = TRUE)
  scale * (r_inv %*% t(r_inv))
}
