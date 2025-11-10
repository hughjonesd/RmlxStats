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

.mlxs_bootstrap_step.default <- function(handler, idx) {
  stop("Bootstrap step not implemented for ", paste(class(handler), collapse = "/"), call. = FALSE)
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
