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
mlxs_boot <- function(
  fun,
  ...,
  B = 200L,
  seed = NULL,
  progress = FALSE,
  compile = FALSE
) {
  if (!is.function(fun)) {
    stop("`fun` must be a function.", call. = FALSE)
  }
  data_list <- list(...)
  if (!length(data_list)) {
    stop(
      "mlxs_boot() requires at least one argument to resample.",
      call. = FALSE
    )
  }

  keep <- vapply(data_list, Negate(is.null), logical(1))
  data_list <- data_list[keep]
  if (!length(data_list)) {
    stop("All supplied arguments are NULL; nothing to resample.", call. = FALSE)
  }

  prepared <- lapply(data_list, .mlxs_boot_prepare_arg)
  dims_first <- vapply(prepared, function(x) Rmlx::mlx_shape(x)[1L], integer(1))
  n_vals <- unique(dims_first)
  if (length(n_vals) != 1L) {
    stop(
      "All arguments must share the same number of rows for mlxs_boot().",
      call. = FALSE
    )
  }
  n_obs <- n_vals

  if (!is.null(seed)) {
    has_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
    old_seed <- if (has_seed) {
      get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
    } else {
      NULL
    }
    on.exit(
      {
        if (has_seed) {
          assign(".Random.seed", old_seed, envir = .GlobalEnv)
        } else if (
          exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
        ) {
          rm(".Random.seed", envir = .GlobalEnv)
        }
      },
      add = TRUE
    )
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
    boot_args <- lapply(prepared, function(x) {
      Rmlx::mlx_gather(x, idx, axes = 1L)
    })
    names(boot_args) <- names(prepared)
    samples[[rep_idx]] <- do.call(fun_eval, boot_args)
    if (!is.null(pb)) {
      utils::setTxtProgressBar(pb, rep_idx)
    }
  }

  list(samples = samples, B = B, seed = seed)
}

#' Bootstrap coefficient summaries for fitted MLXS models
#'
#' Entry point used by model-summary code to compute bootstrap standard errors
#' and percentile intervals. It dispatches to case or residual resampling after
#' validating which combinations are supported for the fitted model type.
#'
#' @param object Fitted `mlxs_lm` or `mlxs_glm` object.
#' @param fit_type Model family for the refit path, `"lm"` or `"glm"`.
#' @param B Number of bootstrap replicates.
#' @param seed Optional random seed.
#' @param progress Logical; show a progress bar.
#' @param method Bootstrap method, `"case"` or `"residual"`.
#' @param level Confidence level for percentile intervals.
#' @return List of bootstrap standard errors, confidence intervals, and
#'   metadata.
#' @noRd
.mlxs_bootstrap_coefs <- function(
  object,
  fit_type = c("lm", "glm"),
  B = 200L,
  seed = NULL,
  progress = FALSE,
  method = c("case", "residual"),
  level = 0.95
) {
  fit_type <- match.arg(fit_type)
  method <- match.arg(method)

  if (method == "residual" && fit_type == "glm") {
    fam <- object$family$family
    if (!fam %in% c("gaussian", "quasigaussian")) {
      stop(
        "Residual bootstrap for mlxs_glm currently supports only gaussian/quasigaussian families.",
        call. = FALSE
      )
    }
  }

  if (method == "case") {
    .mlxs_bootstrap_case(object, fit_type, B, seed, progress, level)
  } else {
    .mlxs_bootstrap_residual(object, B, seed, progress, level)
  }
}

#' Case bootstrap an MLXS fitted model
#'
#' Resamples rows of the original model matrix, response, and optional weights,
#' refits the model on each resample, and forwards the coefficient samples to
#' the summary-statistic helper.
#'
#' @param object Fitted `mlxs_lm` or `mlxs_glm` object.
#' @param fit_type Model family for the refit path, `"lm"` or `"glm"`.
#' @param B Number of bootstrap replicates.
#' @param seed Optional random seed.
#' @param progress Logical; show a progress bar.
#' @param level Confidence level for percentile intervals.
#' @return List from `.mlxs_bootstrap_sample_stats()`.
#' @noRd
.mlxs_bootstrap_case <- function(object, fit_type, B, seed, progress, level) {
  mm <- stats::model.matrix(object$terms, object$model)
  design_mlx <- Rmlx::as_mlx(mm)
  coef_names <- .mlxs_coef_names(object)
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
  
  dtypes <- vapply(boot_res$samples, Rmlx::mlx_dtype, character(1))
  common_dtype <- if (any(dtypes == "float64")) "float64" else "float32"
  common_device <- if (common_dtype == "float64") "cpu" else "gpu"
  Rmlx::local_device(common_device)
  boot_res$samples <- lapply(boot_res$samples, Rmlx::mlx_cast,
                             dtype = common_dtype)

  .mlxs_bootstrap_sample_stats(
    boot_res$samples,
    coef_names,
    B,
    seed,
    method = "case",
    level = level
  )
}

#' Residual bootstrap an MLXS linear model
#'
#' Resamples centered residuals around the fitted values and reuses the original
#' QR decomposition to solve each bootstrap coefficient vector without
#' rebuilding the design matrix.
#'
#' @param object Fitted `mlxs_lm` object or Gaussian `mlxs_glm` object with QR
#'   state.
#' @param B Number of bootstrap replicates.
#' @param seed Optional random seed.
#' @param progress Logical; show a progress bar.
#' @param level Confidence level for percentile intervals.
#' @return List from `.mlxs_bootstrap_sample_stats()`.
#' @noRd
.mlxs_bootstrap_residual <- function(object, B, seed, progress, level) {
  coef_names <- .mlxs_coef_names(object)
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
    Rmlx::mlx_solve_triangular(qr_state$R, qty, upper = TRUE, 
                               device = "cpu")
  }

  boot_res <- mlxs_boot(
    fun = residual_fun,
    residuals = resid_centered,
    B = B,
    seed = seed,
    progress = progress
  )

  .mlxs_bootstrap_sample_stats(
    boot_res$samples,
    coef_names,
    B,
    seed,
    method = "residual",
    level = level
  )
}

#' Summarize bootstrap coefficient samples
#'
#' Aggregates MLX coefficient draws into standard errors and percentile
#' intervals. This is the final shared step for case and residual bootstrap
#' workflows.
#'
#' @param sample_list List of MLX coefficient vectors.
#' @param coef_names Coefficient names.
#' @param B Number of bootstrap replicates.
#' @param seed Optional random seed.
#' @param method Bootstrap method label.
#' @param level Confidence level for percentile intervals.
#' @return List with MLX standard errors, host confidence-interval matrix, and
#'   metadata.
#' @noRd
.mlxs_bootstrap_sample_stats <- function(
  sample_list,
  coef_names,
  B,
  seed,
  method,
  level
) {
  coef_array <- Rmlx::mlx_stack(sample_list, axis = 3L)
  se_mlx <- Rmlx::mlx_std(coef_array, axes = 3L, drop = FALSE, ddof = 1L)
  se_mlx <- Rmlx::mlx_reshape(se_mlx, c(length(coef_names), 1L))
  alpha <- (1 - level) / 2
  confint_mlx <- Rmlx::mlx_quantile(
    coef_array,
    c(alpha, 1 - alpha),
    axis = 3L
  )
  confint_mat <- as.matrix(Rmlx::mlx_reshape(
    confint_mlx,
    c(length(coef_names), 2L)
  ))
  rownames(confint_mat) <- coef_names
  probs <- c(alpha, 1 - alpha) * 100
  colnames(confint_mat) <- paste0(sprintf("%g", probs), " %")
  list(
    se = se_mlx,
    confint = confint_mat,
    samples = NULL,
    B = B,
    seed = seed,
    method = method,
    level = level
  )
}

#' Prepare one argument for MLX bootstrap resampling
#'
#' Normalizes vectors, matrices, and MLX arrays so [mlxs_boot()] can gather rows
#' along the first dimension uniformly.
#'
#' @param x Object to resample.
#' @return MLX array with vectors represented as one-column matrices.
#' @noRd
.mlxs_boot_prepare_arg <- function(x) {
  if (inherits(x, "mlx")) {
    dims <- Rmlx::mlx_shape(x)
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
