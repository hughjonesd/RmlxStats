.mlxs_bootstrap_coefs <- function(object,
                                  fit_type = c("lm", "glm"),
                                  B = 200L,
                                  seed = NULL,
                                  progress = FALSE,
                                  batch_size = 32L,
                                  method = c("case", "residual")) {
  fit_type <- match.arg(fit_type)
  method <- match.arg(method)
  if (method == "residual" && fit_type == "glm") {
    family_name <- object$family$family
    if (!family_name %in% c("gaussian", "quasigaussian")) {
      warning("Residual bootstrap for mlxs_glm currently supported only for gaussian family; falling back to case resampling.", call. = FALSE)
      method <- "case"
    }
  }

  state <- list(
    object = object,
    fit_type = fit_type,
    B = B,
    seed = seed,
    progress = progress
  )
  handler <- .mlxs_bootstrap_method(state, method)
  .mlxs_bootstrap_run(handler)
}

.mlxs_bootstrap_method <- function(state, method) {
  class(state) <- c(sprintf("mlxs_bootstrap_%s", method), "mlxs_bootstrap_state")
  state
}

.mlxs_bootstrap_run <- function(state, ...) {
  UseMethod(".mlxs_bootstrap_run")
}

.mlxs_bootstrap_run.default <- function(state, ...) {
  stop("Unknown bootstrap method: ", paste(class(state), collapse = "/"), call. = FALSE)
}

.mlxs_bootstrap_run.mlxs_bootstrap_case <- function(state, ...) {
  object <- state$object
  fit_type <- state$fit_type
  B <- as.integer(state$B)
  seed <- state$seed
  progress <- state$progress

  state$design_mlx <- Rmlx::as_mlx(stats::model.matrix(object$terms, object$model))
  state$coef_names <- object$coef_names
  state$has_intercept <- any(state$coef_names == "(Intercept)")
  state$y_mlx <- if (fit_type == "glm") object$y else object$residuals + object$fitted.values
  state$weights_mlx <- switch(fit_type, lm = object$weights, glm = object$prior.weights)
  state$coef_init <- object$coefficients
  state$method <- "case"

  dims <- Rmlx::mlx_dim(state$design_mlx)
  coef_stack <- .mlxs_bootstrap_collect(state, n = dims[1L])
  .mlxs_bootstrap_finalize(coef_stack, state$coef_names, state)
}

.mlxs_bootstrap_run.mlxs_bootstrap_residual <- function(state, ...) {
  object <- state$object
  state$qr <- object$qr
  state$coef_names <- object$coef_names
  residuals_mlx <- object$residuals
  fitted_mlx <- object$fitted.values
  state$centered_resid <- residuals_mlx - Rmlx::mlx_mean(residuals_mlx)
  state$fitted_mlx <- fitted_mlx
  state$method <- "residual"
  dims <- Rmlx::mlx_dim(residuals_mlx)
  coef_stack <- .mlxs_bootstrap_collect(state, n = dims[1L])
  .mlxs_bootstrap_finalize(coef_stack, state$coef_names, state)
}

#' @importFrom utils txtProgressBar setTxtProgressBar
#' @importFrom stats sd formula
