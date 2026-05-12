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
#' @param control Optional list of control parameters. Currently supports
#'   `epsilon_f64`, a backward-error threshold for switching to float64 CPU
#'   iterative refinement.
#'
#' @return An object of class `c("mlxs_lm", "mlxs_model")` containing
#'   components similar to an `"lm"` fit, along with MLX intermediates stored in
#'   the `mlx` element.
#'   Note that MLX currently operates in single precision, so fitted values and
#'   diagnostics may differ from `stats::lm()` at around the 1e-6 level. Unlike
#'   [stats::lm()], rank-deficient model matrices are rejected rather than fit
#'   with aliased coefficients.
#' @export
#'
#' @examples
#' fit <- mlxs_lm(mpg ~ cyl + disp, data = mtcars)
#' coef(fit)
mlxs_lm <- function(formula, data, subset, weights, control = list()) {
  call <- match.call()
  control <- do.call(.mlxs_lm_control, control)

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
  .mlxs_check_full_rank(design, "mlxs_lm()")

  weights_mlx <- NULL
  if (!is.null(weights_raw)) {
    weights_mlx <- if (inherits(weights_raw, "mlx")) {
      weights_raw
    } else {
      Rmlx::mlx_matrix(weights_raw, ncol = 1)
    }
    weight_len <- prod(Rmlx::mlx_shape(weights_mlx))
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
    weights = weights_mlx,
    epsilon_f64 = control$epsilon_f64
  )

  result <- list(
    coefficients = fit_res$coefficients,
    fitted.values = fit_res$fitted.values,
    na.action = attr(mf, "na.action"),
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
    assign = assign_vec,
    control = control,
    refined = fit_res$refined,
    refinement_iterations = fit_res$refinement_iterations,
    refinement_initial_error = fit_res$refinement_initial_error,
    refinement_final_error = fit_res$refinement_final_error,
    refinement_delta = fit_res$refinement_delta
  )

  class(result) <- c("mlxs_lm", "mlxs_model")
  result
}

.mlxs_lm_control <- function(epsilon_f64 = 1e-6) {
  if (!is.null(epsilon_f64)) {
    if (!is.numeric(epsilon_f64) || length(epsilon_f64) != 1L ||
        !is.finite(epsilon_f64) || epsilon_f64 <= 0) {
      stop("epsilon_f64 must be a positive finite number.", call. = FALSE)
    }
  }
  list(epsilon_f64 = epsilon_f64)
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
#' @param epsilon_f64 Optional backward-error threshold for switching to
#'   float64 CPU iterative refinement. When `NULL`, no refinement is run.
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
mlxs_lm_fit <- function(x, y, weights = NULL, epsilon_f64 = NULL) {
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
    dims <- Rmlx::mlx_shape(x_orig)
    w_broadcast <- Rmlx::mlx_broadcast_to(w_sqrt, dims)
    x_work <- x_orig * w_broadcast
    y_work <- y_orig * w_sqrt
  }

  # qr has to be on cpu at present...
  qr_fit <- qr(x_work, device = "cpu")
  qty_mlx <- crossprod(qr_fit$Q, y_work)
  # so does solve_triangular 
  coef_mlx <- Rmlx::mlx_solve_triangular(qr_fit$R, qty_mlx, upper = TRUE,
                                         device = "cpu")
  fitted_mlx <- x_orig %*% coef_mlx
  residual_mlx <- y_orig - fitted_mlx
  refined <- FALSE
  refinement_iterations <- 0L
  refinement_initial_error <- NA_real_
  refinement_final_error <- NA_real_
  refinement_delta <- NA_real_

  if (!is.null(epsilon_f64) &&
      !identical(Rmlx::mlx_dtype(coef_mlx), "float64")) {
    fitted_work <- x_work %*% coef_mlx
    residual_work <- y_work - fitted_work
    refinement_initial_error <- .mlxs_lm_backward_error(
      x_work,
      residual_work,
      y_work,
      fitted_work
    )
    refinement_final_error <- refinement_initial_error

    if (refinement_initial_error > epsilon_f64) {
      Rmlx::local_default_device("cpu")
      x_work_64 <- Rmlx::mlx_cast(x_work, dtype = "float64", device = "cpu")
      y_work_64 <- Rmlx::mlx_cast(y_work, dtype = "float64", device = "cpu")
      x_orig_64 <- Rmlx::mlx_cast(x_orig, dtype = "float64", device = "cpu")
      y_orig_64 <- Rmlx::mlx_cast(y_orig, dtype = "float64", device = "cpu")
      coef_64 <- Rmlx::mlx_cast(coef_mlx, dtype = "float64", device = "cpu")
      qr_64 <- list(
        Q = Rmlx::mlx_cast(qr_fit$Q, dtype = "float64", device = "cpu"),
        R = Rmlx::mlx_cast(qr_fit$R, dtype = "float64", device = "cpu")
      )
      class(qr_64) <- c("mlx_qr", "list")

      previous_error <- refinement_initial_error
      maxit <- 3L
      for (iter in seq_len(maxit)) {
        fitted_work <- x_work_64 %*% coef_64
        residual_work <- y_work_64 - fitted_work
        delta_mlx <- Rmlx::mlx_solve_triangular(
          qr_64$R,
          crossprod(qr_64$Q, residual_work),
          upper = TRUE,
          device = "cpu"
        )
        coef_candidate <- coef_64 + delta_mlx
        candidate_fitted <- x_work_64 %*% coef_candidate
        candidate_error <- .mlxs_lm_backward_error(
          x_work_64,
          y_work_64 - candidate_fitted,
          y_work_64,
          candidate_fitted
        )

        refinement_delta <- as.numeric(max(abs(delta_mlx)))
        if (!is.finite(candidate_error) || candidate_error >= previous_error) {
          break
        }

        coef_64 <- coef_candidate
        refinement_iterations <- iter
        refinement_final_error <- candidate_error
        refined <- TRUE
        previous_error <- candidate_error

        if (candidate_error <= epsilon_f64) {
          break
        }
      }

      if (refined) {
        coef_mlx <- coef_64
        qr_fit <- qr_64
        qty_mlx <- crossprod(qr_fit$Q, y_work_64)
        fitted_mlx <- x_orig_64 %*% coef_mlx
        residual_mlx <- y_orig_64 - fitted_mlx
      }
    }
  }

  list(
    coefficients = coef_mlx,
    fitted.values = fitted_mlx,
    residuals = residual_mlx,
    effects = qty_mlx,
    qr = qr_fit,
    refined = refined,
    refinement_iterations = refinement_iterations,
    refinement_initial_error = refinement_initial_error,
    refinement_final_error = refinement_final_error,
    refinement_delta = refinement_delta
  )
}

.mlxs_lm_backward_error <- function(x, residual, y, fitted) {
  gradient <- crossprod(x, residual)
  scale <- crossprod(abs(x), abs(y) + abs(fitted))
  scaled <- abs(gradient) / Rmlx::mlx_clip(scale, min = .Machine$double.eps)
  as.numeric(max(scaled))
}
