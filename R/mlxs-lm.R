#' MLX-backed linear regression
#'
#' Fit a linear model via QR decomposition using MLX arrays on Apple Silicon
#' devices. The interface mirrors [stats::lm()] for the common arguments.
#'
#' @param formula Model formula.
#' @param data Optional data frame, tibble, or environment containing the
#'   variables in the model.
#' @param subset Optional expression for subsetting observations.
#'
#' @return An object of class `c("mlxs_lm", "mlxs_model")` containing
#'   components similar to an `"lm"` fit, along with MLX intermediates stored in
#'   the `mlx` element.
#' @export
#'
#' @examples
#' if (requireNamespace("Rmlx", quietly = TRUE)) {
#'   fit <- mlxs_lm(mpg ~ cyl + disp, data = mtcars)
#'   coef(fit)
#' }
mlxs_lm <- function(formula, data, subset) {
  call <- match.call()

  mf <- match.call(expand.dots = FALSE)
  arg_names <- c("formula", "data", "subset")
  keep <- match(arg_names, names(mf), nomatch = 0L)
  mf <- mf[c(1L, keep)]
  mf[[1L]] <- quote(model.frame)
  mf <- eval(mf, parent.frame())

  terms <- attr(mf, "terms")
  response <- model.response(mf)
  if (is.matrix(response)) {
    response <- drop(response)
  }

  design <- model.matrix(terms, mf)

  n_obs <- nrow(design)
  if (is.null(n_obs) || n_obs == 0L) {
    stop("No observations after processing model frame.", call. = FALSE)
  }

  n_coef <- ncol(design)
  if (is.null(n_coef) || n_coef == 0L) {
    stop("No coefficients to estimate; provide predictors in the formula.", call. = FALSE)
  }

  x_mlx <- Rmlx::as_mlx(design)
  y_mlx <- Rmlx::as_mlx(matrix(response, ncol = 1))

  qr_fit <- qr(x_mlx)
  qty <- crossprod(qr_fit$Q, y_mlx)
  coef_mlx <- Rmlx::mlx_solve_triangular(qr_fit$R, qty, upper = TRUE)

  fitted_mlx <- x_mlx %*% coef_mlx
  residual_mlx <- y_mlx - fitted_mlx

  coefficients <- drop(as.matrix(coef_mlx))
  names(coefficients) <- colnames(design)

  fitted_values <- drop(as.matrix(fitted_mlx))
  residuals <- drop(as.matrix(residual_mlx))
  effects <- drop(as.matrix(qty))
  names(effects) <- colnames(design)

  if (!is.null(rownames(design))) {
    names(fitted_values) <- rownames(design)
    names(residuals) <- rownames(design)
  }

  result <- list(
    coefficients = coefficients,
    residuals = residuals,
    effects = effects,
    fitted.values = fitted_values,
    rank = n_coef,
    df.residual = n_obs - n_coef,
    call = call,
    terms = terms,
    model = mf,
    mlx = list(
      qr = qr_fit,
      x = x_mlx,
      y = y_mlx
    )
  )

  class(result) <- c("mlxs_lm", "mlxs_model")
  result
}
