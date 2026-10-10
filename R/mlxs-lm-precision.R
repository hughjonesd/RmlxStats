#' Check the numerical precision of an MLX linear model
#'
#' Evaluate the residual equations in CPU float64 using the original model
#' frame, the fitted coefficients, and the stored QR factor. No second QR
#' factorization is needed. This diagnostic is optional and makes an
#' additional pass over the design matrix.
#'
#' @param object An `mlxs_lm` fit.
#' @param tol Warning threshold in joint standard-error units. The default
#'   warns when the estimated numerical displacement exceeds one unit.
#'   Use `Inf` to obtain the diagnostic without a finite-error warning.
#'
#' @return A numeric scalar estimating the joint numerical coefficient error
#'   in standard-error units. Large values issue a warning.
#'
#' @details
#' The diagnostic computes `||R^-T X' r|| / sigma`, with square-root weights
#' applied to `X` and `r` for weighted fits. With an accurate QR factor this
#' estimates the distance from the least-squares solution measured using its
#' coefficient covariance. It bounds individual coefficient errors in their
#' standard-error units, but can be larger because it combines all directions.
#'
#' The computation uses float64 to avoid masking float32 solve errors with
#' another float32 residual calculation. It checks the supplied model data;
#' it cannot recover precision already lost before those data were supplied.
#' An inaccurate QR factor can make the estimate unreliable, so a small value
#' is not a guarantee of accuracy. To investigate a warning, refit the
#' original data as float64 arrays using [mlxs_lm_fit()] on the CPU.
#'
#' @seealso [mlxs_lm()], [mlxs_lm_fit()]
#' @export
#' @examples
#' fit <- mlxs_lm(mpg ~ cyl + disp, data = mtcars)
#' mlxs_lm_precision(fit)
mlxs_lm_precision <- function(object, tol = 1) {
  if (!inherits(object, "mlxs_lm")) {
    stop("'object' must be an mlxs_lm fit.", call. = FALSE)
  }
  if (!is.numeric(tol) || length(tol) != 1L || is.na(tol) || tol < 0) {
    stop("'tol' must be a non-negative number.", call. = FALSE)
  }
  if (object$df.residual <= 0) {
    stop("Precision checks require positive residual degrees of freedom.",
         call. = FALSE)
  }
  Rmlx::local_device("cpu")
  x <- Rmlx::as_mlx(model.matrix(
    object$terms, object$model, contrasts.arg = object$contrasts
  ), "float64")
  y <- Rmlx::as_mlx(model.response(object$model), "float64")
  y <- Rmlx::mlx_reshape(y, c(nrow(x), 1L))
  beta <- Rmlx::as_mlx(object$coefficients, "float64")
  r <- Rmlx::as_mlx(object$qr$R, "float64")
  residual <- y - x %*% beta
  if (!is.null(object$weights)) {
    w <- Rmlx::as_mlx(object$model[["(weights)"]], "float64")
    w <- Rmlx::mlx_reshape(w, c(nrow(x), 1L))
    x <- x * sqrt(w)
    residual <- residual * sqrt(w)
  }
  qty_error <- Rmlx::mlx_solve_triangular(
    t(r), crossprod(x, residual), upper = FALSE, device = "cpu"
  )
  error_se <- as.numeric(sqrt(sum(qty_error^2) * object$df.residual /
                               sum(residual^2)))
  if (!is.finite(error_se) || error_se > tol) {
    warning(
      "Numerical coefficient error may exceed ", tol,
      " joint standard-error units; refit original data in CPU float64.",
      call. = FALSE
    )
  }
  error_se
}
