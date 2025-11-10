.mlxs_bootstrap_coefs <- function(object,
                                  fit_type = c("lm", "glm"),
                                  B = 200L,
                                  seed = NULL,
                                  progress = FALSE,
                                  batch_size = 32L,
                                  method = c("case", "residual")) {
  fit_type <- match.arg(fit_type)
  method <- match.arg(method)
  if (method == "residual" && fit_type == "glm" && !object$family$family %in% c("gaussian", "quasigaussian")) {
    stop("Residual bootstrap for mlxs_glm currently supports only gaussian/quasigaussian families.", call. = FALSE)
  }

  handler <- structure(
    list(
      object = object,
      fit_type = fit_type,
      B = B,
      seed = seed,
      progress = progress,
      method = method
    ),
    class = sprintf("mlxs_bootstrap_%s", method)
  )
  .mlxs_bootstrap_run(handler)
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

# helpers ---------------------------------------------------------------

.mlxs_bootstrap_collect <- function(handler, n) {
  B <- handler$B
  seed <- handler$seed
  progress <- handler$progress
  if (!is.null(seed)) {
    old_seed <- .Random.seed
    on.exit(assign(".Random.seed", old_seed, envir = .GlobalEnv), add = TRUE)
    set.seed(seed)
  }
  coef_stack <- vector("list", B)
  pb <- NULL
  if (isTRUE(progress)) {
    pb <- utils::txtProgressBar(min = 0, max = B, style = 3)
    on.exit(close(pb), add = TRUE)
  }
  for (rep_idx in seq_len(B)) {
    idx <- sample.int(n, n, replace = TRUE)
    coef_stack[[rep_idx]] <- .mlxs_bootstrap_step(handler, idx)
    if (!is.null(pb)) {
      utils::setTxtProgressBar(pb, rep_idx)
    }
  }
  coef_stack
}

.mlxs_bootstrap_finalize <- function(coef_stack, coef_names, handler) {
  coef_array <- Rmlx::mlx_stack(coef_stack, axis = 3L)
  se_mlx <- Rmlx::mlx_std(coef_array, axis = 3L, drop = FALSE, ddof = 1L)
  se_mlx <- Rmlx::mlx_reshape(se_mlx, c(length(coef_names), 1L))
  list(se = se_mlx, samples = NULL, B = handler$B, seed = handler$seed, method = handler$method)
}

.mlxs_bootstrap_step <- function(handler, idx) {
  UseMethod(".mlxs_bootstrap_step")
}

.mlxs_bootstrap_step.mlxs_bootstrap_case <- function(handler, idx) {
  x_boot <- handler$design_mlx[idx, , drop = FALSE]
  y_boot <- handler$y_mlx[idx, , drop = FALSE]
  w_boot <- if (is.null(handler$weights_mlx)) NULL else handler$weights_mlx[idx, , drop = FALSE]
  if (handler$fit_type == "lm") {
    mlxs_lm_fit(x_boot, y_boot, weights = w_boot)$coefficients
  } else {
    .mlxs_glm_fit_core(
      design = x_boot,
      response = y_boot,
      weights_raw = w_boot,
      family = handler$object$family,
      control = handler$object$control,
      coef_start = handler$coef_init,
      coef_names = handler$coef_names,
      has_intercept = handler$has_intercept
    )$coefficients
  }
}

.mlxs_bootstrap_step.mlxs_bootstrap_residual <- function(handler, idx) {
  resid_draw <- handler$centered_resid[idx, , drop = FALSE]
  y_boot <- handler$fitted_mlx + resid_draw
  qty <- crossprod(handler$qr$Q, y_boot)
  Rmlx::mlx_solve_triangular(handler$qr$R, qty, upper = TRUE)
}

#' @importFrom utils txtProgressBar setTxtProgressBar
#' @importFrom stats sd formula
