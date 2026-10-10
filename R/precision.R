#' Validate the precision threshold shared by the two formula interfaces
#' @noRd
.mlxs_check_precision_tol <- function(tol) {
  if (!is.null(tol) &&
      (!is.numeric(tol) || length(tol) != 1L || is.na(tol) || tol < 0)) {
    stop("'precision_tol' must be NULL or a non-negative number.",
         call. = FALSE)
  }
  tol
}

#' Estimate stationarity error using original data and the stored QR factor
#'
#' For GLMs this is a local scoring-step estimate using the last IRLS factor.
#' NULL avoids the entire data pass.
#' @noRd
.mlxs_precision_check <- function(object, design, response, weights, tol) {
  if (is.null(tol)) return(NULL)
  if (object$df.residual <= 0) return(NA_real_)
  Rmlx::local_device("cpu")
  x <- Rmlx::as_mlx(design, "float64")
  y <- Rmlx::as_mlx(response, "float64")
  y <- Rmlx::mlx_reshape(y, c(nrow(x), 1L))
  beta <- Rmlx::as_mlx(object$coefficients, "float64")
  r <- Rmlx::as_mlx(object$qr$R, "float64")
  w <- if (is.null(weights)) {
    Rmlx::mlx_full(c(nrow(x), 1L), 1, dtype = "float64")
  } else {
    Rmlx::mlx_reshape(Rmlx::as_mlx(
      weights, "float64"
    ), c(nrow(x), 1L))
  }
  eta <- x %*% beta
  if (inherits(object, "mlxs_glm")) {
    mu <- object$family$linkinv(eta)
    mu <- .mlxs_glm_clamp_mu(mu, object$family)
    residual <- y - mu
    score_residual <- w * residual * object$family$mu.eta(eta) /
      object$family$variance(mu)
    dispersion <- object$dispersion
    if (identical(object$family$family, "gaussian")) {
      dispersion <- as.numeric(sum(w * residual^2) / object$df.residual)
    }
    warning_text <- "Numerical error or incomplete GLM convergence"
    warning_action <-
      "consider a float64 refit or tighter convergence controls."
  } else {
    residual <- y - eta
    score_residual <- w * residual
    dispersion <- as.numeric(sum(w * residual^2) / object$df.residual)
    warning_text <- "Numerical coefficient error"
    warning_action <- "refit the original data in float64."
  }
  # An exact fit has no residual scale in which to express the error.
  if (is.finite(dispersion) && dispersion == 0) return(NA_real_)
  qty_error <- Rmlx::mlx_solve_triangular(
    t(r), crossprod(x, score_residual), upper = FALSE, device = "cpu"
  )
  error_se <- as.numeric(sqrt(sum(qty_error^2) / dispersion))
  if (!is.finite(error_se) || error_se > tol) {
    warning(
      warning_text, " may exceed precision_tol = ", tol,
      " joint standard-error units (estimated ", signif(error_se, 3),
      "); ", warning_action,
      call. = FALSE
    )
  }
  error_se
}
