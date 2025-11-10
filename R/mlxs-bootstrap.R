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

  design_mat <- stats::model.matrix(object$terms, object$model)
  coef_names <- object$coef_names

  design_mlx <- Rmlx::as_mlx(design_mat)
  dims <- Rmlx::mlx_dim(design_mlx)
  n <- dims[1L]
  has_intercept <- any(coef_names == "(Intercept)")

  y_mlx <- if (fit_type == "glm") {
    object$y
  } else {
    object$residuals + object$fitted.values
  }

  weights_mlx <- switch(
    fit_type,
    lm = object$weights,
    glm = object$prior.weights
  )

  coef_init <- object$coefficients

  if (!is.null(seed)) {
    old_seed <- .Random.seed
    on.exit(assign(".Random.seed", old_seed, envir = .GlobalEnv), add = TRUE)
    set.seed(seed)
  }

  pb <- NULL
  if (isTRUE(progress)) {
    pb <- utils::txtProgressBar(min = 0, max = B, style = 3)
    on.exit(close(pb), add = TRUE)
  }

  coef_stack <- .mlxs_bootstrap_collect(
    B = B,
    n = n,
    seed = seed,
    progress = progress,
    build_boot = function(idx) {
      x_boot <- design_mlx[idx, , drop = FALSE]
      y_boot <- y_mlx[idx, , drop = FALSE]
      w_boot <- if (is.null(weights_mlx)) NULL else weights_mlx[idx, , drop = FALSE]
      if (fit_type == "lm") {
        mlxs_lm_fit(x_boot, y_boot, weights = w_boot)$coefficients
      } else {
        .mlxs_glm_fit_core(
          design = x_boot,
          response = y_boot,
          weights_raw = w_boot,
          family = object$family,
          control = object$control,
          coef_start = coef_init,
          coef_names = coef_names,
          has_intercept = has_intercept
        )$coefficients
      }
    }
  )
  .mlxs_bootstrap_finalize(coef_stack, coef_names, method = "case", B = B, seed = seed)
}

.mlxs_bootstrap_run.mlxs_bootstrap_residual <- function(state, ...) {
  object <- state$object
  fit_type <- state$fit_type
  B <- as.integer(state$B)
  seed <- state$seed
  progress <- state$progress

  qr_fit <- object$qr
  coef_names <- object$coef_names
  residuals_mlx <- object$residuals
  fitted_mlx <- object$fitted.values
  dims <- Rmlx::mlx_dim(residuals_mlx)
  n <- dims[1L]
  mean_resid <- Rmlx::mlx_mean(residuals_mlx)
  centered_resid <- residuals_mlx - mean_resid

  coef_stack <- .mlxs_bootstrap_collect(
    B = B,
    n = n,
    seed = seed,
    progress = progress,
    build_boot = function(idx) {
      resid_draw <- centered_resid[idx, , drop = FALSE]
      y_boot <- fitted_mlx + resid_draw
      qty <- crossprod(qr_fit$Q, y_boot)
      Rmlx::mlx_solve_triangular(qr_fit$R, qty, upper = TRUE)
    }
  )
  .mlxs_bootstrap_finalize(coef_stack, coef_names, method = "residual", B = B, seed = seed)
}

#' @importFrom utils txtProgressBar setTxtProgressBar
#' @importFrom stats sd formula
