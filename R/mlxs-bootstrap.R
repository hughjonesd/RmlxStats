#' Bootstrap MLX arrays along the first dimension
#'
#' @description
#' `mlxs_boot()` resamples observations from one or more MLX arrays, calls a
#' user-supplied function on each resampled batch, and returns the collected
#' results. Every argument supplied via `...` must share the same size in its
#' first dimension (number of observations). Arguments that do not need
#' resampling should be captured in the environment of `fun` instead of being
#' passed through `...`.
#'
#' @param fun Function called on each bootstrap draw. It must accept the same
#'   named arguments as supplied through `...`.
#' @param ... Arrays, matrices, or vectors that should be resampled along the
#'   first dimension before being passed to `fun`.
#' @param B Number of bootstrap iterations.
#' @param seed Optional integer seed for reproducibility.
#' @param progress Logical; if `TRUE`, show a text progress bar.
#' @param compile Logical; compile `fun` once via [Rmlx::mlx_compile()] before
#'   entering the resampling loop. Defaults to `FALSE`.
#'
#' @return A list with elements `samples` (the raw results from `fun`), `B`, and
#'   `seed`.
#' @export
#' @importFrom utils txtProgressBar setTxtProgressBar
mlxs_boot <- function(fun, ..., B = 200L, seed = NULL, progress = FALSE,
                      compile = FALSE) {
  if (!is.function(fun)) {
    stop("`fun` must be a function.", call. = FALSE)
  }
  data_list <- list(...)
  if (!length(data_list)) {
    stop("mlxs_boot() requires at least one argument to resample.", call. = FALSE)
  }

  keep <- vapply(data_list, Negate(is.null), logical(1))
  data_list <- data_list[keep]
  if (!length(data_list)) {
    stop("All supplied arguments are NULL; nothing to resample.", call. = FALSE)
  }

  prepared <- lapply(data_list, .mlxs_boot_prepare_arg)
  dims_first <- vapply(prepared, function(x) Rmlx::mlx_dim(x)[1L], integer(1))
  n_vals <- unique(dims_first)
  if (length(n_vals) != 1L) {
    stop("All arguments must share the same number of rows for mlxs_boot().", call. = FALSE)
  }
  n_obs <- n_vals

  if (!is.null(seed)) {
    has_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
    old_seed <- if (has_seed) get(".Random.seed", envir = .GlobalEnv, inherits = FALSE) else NULL
    on.exit({
      if (has_seed) {
        assign(".Random.seed", old_seed, envir = .GlobalEnv)
      } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
        rm(".Random.seed", envir = .GlobalEnv)
      }
    }, add = TRUE)
    set.seed(seed)
  }

  B <- as.integer(B)
  if (B <= 0) {
    stop("`B` must be a positive integer.", call. = FALSE)
  }

  samples <- vector("list", B)
  pb <- NULL
  if (isTRUE(progress)) {
    pb <- utils::txtProgressBar(min = 0, max = B, style = 3)
    on.exit(close(pb), add = TRUE)
  }

  fun_eval <- fun
  if (isTRUE(compile)) {
    fun_eval <- Rmlx::mlx_compile(fun)
  }

  for (rep_idx in seq_len(B)) {
    idx <- sample.int(n_obs, n_obs, replace = TRUE)
    boot_args <- lapply(prepared, .mlxs_boot_take, idx = idx)
    names(boot_args) <- names(prepared)
    samples[[rep_idx]] <- do.call(fun_eval, boot_args)
    if (!is.null(pb)) {
      utils::setTxtProgressBar(pb, rep_idx)
    }
  }

  list(samples = samples, B = B, seed = seed)
}

.mlxs_bootstrap_coefs <- function(object,
                                  fit_type = c("lm", "glm"),
                                  B = 200L,
                                  seed = NULL,
                                  progress = FALSE,
                                  method = c("case", "residual")) {
  fit_type <- match.arg(fit_type)
  method <- match.arg(method)

  if (method == "residual" && fit_type == "glm") {
    fam <- object$family$family
    if (!fam %in% c("gaussian", "quasigaussian")) {
      stop("Residual bootstrap for mlxs_glm currently supports only gaussian/quasigaussian families.", call. = FALSE)
    }
  }

  if (method == "case") {
    return(.mlxs_bootstrap_case(object, fit_type, B, seed, progress))
  }
  .mlxs_bootstrap_residual(object, B, seed, progress)
}

.mlxs_bootstrap_case <- function(object, fit_type, B, seed, progress) {
  mm <- stats::model.matrix(object$terms, object$model)
  design_mlx <- Rmlx::as_mlx(mm)
  coef_names <- object$coef_names
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

  case_fun <- if (fit_type == "lm") {
    function(X, y, weights = NULL) {
      mlxs_lm_fit(X, y, weights = weights)$coefficients
    }
  } else {
    family <- object$family
    control <- object$control
    coef_start <- object$coefficients
    has_intercept <- any(coef_names == "(Intercept)")
    function(X, y, weights) {
      .mlxs_glm_fit_core(
        design = X,
        response = y,
        weights_raw = weights,
        family = family,
        control = control,
        coef_start = coef_start,
        coef_names = coef_names,
        has_intercept = has_intercept
      )$coefficients
    }
  }

  boot_args <- list(X = design_mlx, y = y_mlx)
  if (!is.null(weights_mlx)) {
    boot_args$weights <- weights_mlx
  }

  boot_res <- do.call(
    mlxs_boot,
    c(list(fun = case_fun, B = B, seed = seed, progress = progress), boot_args)
  )

  .mlxs_bootstrap_from_samples(boot_res$samples, coef_names, B, seed, method = "case")
}

.mlxs_bootstrap_residual <- function(object, B, seed, progress) {
  coef_names <- object$coef_names
  residuals_mlx <- object$residuals
  resid_centered <- residuals_mlx - Rmlx::mlx_mean(residuals_mlx)
  fitted_mlx <- object$fitted.values
  qr_state <- object$qr

  if (is.null(qr_state$Q) || is.null(qr_state$R)) {
    stop("QR decomposition is required for residual bootstrap.", call. = FALSE)
  }

  residual_fun <- function(residuals) {
    y_boot <- fitted_mlx + residuals
    qty <- crossprod(qr_state$Q, y_boot)
    Rmlx::mlx_solve_triangular(qr_state$R, qty, upper = TRUE)
  }

  boot_res <- mlxs_boot(
    fun = residual_fun,
    residuals = resid_centered,
    B = B,
    seed = seed,
    progress = progress
  )

  .mlxs_bootstrap_from_samples(boot_res$samples, coef_names, B, seed, method = "residual")
}

.mlxs_bootstrap_from_samples <- function(sample_list, coef_names, B, seed, method) {
  coef_array <- Rmlx::mlx_stack(sample_list, axis = 3L)
  se_mlx <- Rmlx::mlx_std(coef_array, axis = 3L, drop = FALSE, ddof = 1L)
  se_mlx <- Rmlx::mlx_reshape(se_mlx, c(length(coef_names), 1L))
  list(se = se_mlx, samples = NULL, B = B, seed = seed, method = method)
}

.mlxs_boot_prepare_arg <- function(x) {
  if (inherits(x, "mlx")) {
    dims <- Rmlx::mlx_dim(x)
    if (length(dims) == 1L) {
      return(Rmlx::mlx_reshape(x, c(dims[1L], 1L)))
    }
    return(x)
  }
  if (is.vector(x) && !is.list(x)) {
    return(Rmlx::mlx_matrix(x, ncol = 1))
  }
  Rmlx::as_mlx(x)
}

.mlxs_boot_take <- function(x, idx) {
  dims <- Rmlx::mlx_dim(x)
  nd <- length(dims)
  if (nd == 1L) {
    return(x[idx, drop = FALSE])
  }
  if (nd == 2L) {
    return(x[idx, , drop = FALSE])
  }
  if (nd == 3L) {
    return(x[idx, , , drop = FALSE])
  }
  # Fallback: build a call to `[` with explicit empty arguments for higher dims
  subs <- vector("list", nd)
  subs[[1L]] <- idx
  if (nd > 1L) {
    for (i in 2:nd) {
      subs[[i]] <- quote(expr = )
    }
  }
  subs$drop <- FALSE
  do.call(`[`, c(list(x), subs))
}
