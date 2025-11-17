#' MLX-backed linear regression
#'
#' Fit a linear model via QR decomposition using MLX arrays on Apple Silicon
#' devices. The interface mirrors [stats::lm()] for the common arguments.
#'
#' @param formula Model formula.
#' @param data Optional data frame, tibble, or environment containing the
#'   variables in the model.
#' @param subset Optional expression for subsetting observations.
#' @param weights Optional non-negative observation weights. Treated like the
#'   `weights` argument to [stats::lm()], i.e. they enter the fit via weighted
#'   least squares.
#'
#' @return An object of class `c("mlxs_lm", "mlxs_model")` containing
#'   components similar to an `"lm"` fit, along with MLX intermediates stored in
#'   the `mlx` element.
#'   Note that MLX currently operates in single precision, so fitted values and
#'   diagnostics may differ from `stats::lm()` at around the 1e-6 level.
#' @export
#'
#' @examples
#' fit <- mlxs_lm(mpg ~ cyl + disp, data = mtcars)
#' coef(fit)
mlxs_lm <- function(formula, data, subset, weights) {
  call <- match.call()

  mf <- match.call(expand.dots = FALSE)
  arg_names <- c("formula", "data", "subset", "weights")
  keep <- match(arg_names, names(mf), nomatch = 0L)
  mf <- mf[c(1L, keep)]
  mf[[1L]] <- quote(model.frame)
  mf <- eval(mf, parent.frame())

  terms <- attr(mf, "terms")
  response <- model.response(mf)
  if (is.matrix(response) && ncol(response) == 1L) {
    response <- drop(response)
  }
  design <- model.matrix(terms, mf)
  assign_vec <- attr(design, "assign")
  weights_raw <- mf[["(weights)", exact = TRUE]]

  n_obs <- nrow(design)
  if (is.null(n_obs) || n_obs == 0L) {
    stop("No observations after processing model frame.", call. = FALSE)
  }

  n_coef <- ncol(design)
  if (is.null(n_coef) || n_coef == 0L) {
    stop(
      "No coefficients to estimate; provide predictors in the formula.",
      call. = FALSE
    )
  }

  weights_mlx <- NULL
  if (!is.null(weights_raw)) {
    weights_mlx <- if (inherits(weights_raw, "mlx")) {
      weights_raw
    } else {
      Rmlx::mlx_matrix(weights_raw, ncol = 1)
    }
    weight_len <- prod(Rmlx::mlx_dim(weights_mlx))
    if (weight_len != n_obs) {
      stop(
        "Length of 'weights' must match number of observations.",
        call. = FALSE
      )
    }
    if (any(!Rmlx::mlx_isfinite(weights_mlx))) {
      stop("Weights must be non-negative and finite.", call. = FALSE)
    }
    if (any(weights_mlx < 0)) {
      stop("Weights must be non-negative and finite.", call. = FALSE)
    }
  }

  design_mlx <- Rmlx::as_mlx(design)
  response_mlx <- if (inherits(response, "mlx")) {
    response
  } else {
    Rmlx::mlx_matrix(response, ncol = 1)
  }

  fit_res <- mlxs_lm_fit(
    x = design_mlx,
    y = response_mlx,
    weights = weights_mlx
  )

  result <- list(
    coefficients = fit_res$coefficients,
    fitted.values = fit_res$fitted.values,
    residuals = fit_res$residuals,
    effects = fit_res$effects,
    rank = n_coef,
    df.residual = n_obs - n_coef,
    call = call,
    terms = terms,
    model = mf,
    qr = fit_res$qr,
    coef_names = colnames(design),
    weights = weights_mlx,
    assign = assign_vec
  )

  class(result) <- c("mlxs_lm", "mlxs_model")
  result
}

#' Fit an MLX linear model from design matrices
#'
#' @description
#' `mlxs_lm_fit()` powers [mlxs_lm()] by wrapping the QR-based solver that runs
#' entirely on MLX arrays.
#'
#' @param x MLX design matrix (or object coercible via [Rmlx::as_mlx()]) whose
#'   rows represent observations and columns represent predictors.
#' @param y MLX column vector (or object coercible via [Rmlx::as_mlx()]) holding
#'   the response values.
#' @param weights Optional MLX column vector or numeric vector of non-negative
#'   observation weights. When supplied, weighted least squares are fit via the
#'   standard square-root weighting.
#'
#' @return A list with components `coefficients`, `fitted.values`, `residuals`,
#'   `effects`, and `qr`, mirroring the corresponding pieces of [stats::lm()].
#'   Array-valued components remain MLX matrices to keep downstream GPU
#'   pipelines in device memory.
#'
#' @details
#' Inputs that are not already MLX objects are converted with
#' [Rmlx::as_mlx()] or [Rmlx::mlx_matrix()] so callers can provide base-R
#' matrices or vectors. Weighted fits are performed by applying the standard
#' square-root weight transform before solving the QR system.
#'
#' @examples
#' x <- Rmlx::as_mlx(cbind(1, as.matrix(mtcars[c("cyl", "disp")])))
#' y <- Rmlx::mlx_matrix(mtcars$mpg, ncol = 1)
#' fit <- mlxs_lm_fit(x, y)
#' drop(as.matrix(fit$coefficients))
#'
#' @export
mlxs_lm_fit <- function(x, y, weights = NULL) {
  x_orig <- Rmlx::as_mlx(x)
  y_orig <- if (inherits(y, "mlx")) y else Rmlx::mlx_matrix(y, ncol = 1)

  x_work <- x_orig
  y_work <- y_orig
  if (!is.null(weights)) {
    w_col <- if (inherits(weights, "mlx")) {
      weights
    } else {
      Rmlx::mlx_matrix(weights, ncol = 1)
    }
    w_sqrt <- sqrt(w_col)
    dims <- Rmlx::mlx_dim(x_orig)
    w_broadcast <- Rmlx::mlx_broadcast_to(w_sqrt, dims)
    x_work <- x_orig * w_broadcast
    y_work <- y_orig * w_sqrt
  }

  qr_fit <- qr(x_work)
  qty_mlx <- crossprod(qr_fit$Q, y_work)
  coef_mlx <- Rmlx::mlx_solve_triangular(qr_fit$R, qty_mlx, upper = TRUE)

  fitted_mlx <- x_orig %*% coef_mlx
  residual_mlx <- y_orig - fitted_mlx

  list(
    coefficients = coef_mlx,
    fitted.values = fitted_mlx,
    residuals = residual_mlx,
    effects = qty_mlx,
    qr = qr_fit
  )
}
