.mlxs_bootstrap_coefs <- function(object,
                                  fit_type = c("lm", "glm"),
                                  B = 200L,
                                  seed = NULL,
                                  progress = FALSE,
                                  batch_size = 32L,
                                  method = c("case", "residual")) {
  fit_type <- match.arg(fit_type)
  method <- match.arg(method)
  if (method == "residual") {
    if (fit_type == "lm") {
      return(.mlxs_bootstrap_residual_lm(object, B = B, seed = seed, progress = progress, batch_size = batch_size))
    }
    if (fit_type == "glm") {
      family_name <- object$family$family
      if (!family_name %in% c("gaussian", "quasigaussian")) {
        warning("Residual bootstrap for mlxs_glm currently supported only for gaussian family; using case resampling.", call. = FALSE)
        method <- "case"
      } else {
        return(.mlxs_bootstrap_residual_glm(object, B = B, seed = seed, progress = progress, batch_size = batch_size))
      }
    }
  }
  if (is.null(object$model)) {
    stop("Bootstrap requires the original model frame to be stored.", call. = FALSE)
  }
  design_mat <- stats::model.matrix(object$terms, object$model)
  n <- nrow(design_mat)
  if (is.null(n) || n == 0L) {
    stop("Design matrix not available for bootstrap.", call. = FALSE)
  }
  coef_names <- object$coef_names
  if (is.null(coef_names)) {
    stop("Model must have named coefficients for bootstrap.", call. = FALSE)
  }
  response_num <- if (fit_type == "glm") {
    .mlxs_as_numeric(object$y)
  } else {
    .mlxs_as_numeric(object$residuals + object$fitted.values)
  }
  if (length(response_num) != n) {
    stop("Response length mismatch during bootstrap.", call. = FALSE)
  }
  weights_num <- if (!is.null(object$prior.weights)) .mlxs_as_numeric(object$prior.weights) else NULL
  if (!is.null(weights_num) && length(weights_num) != n) {
    stop("Weight length mismatch during bootstrap.", call. = FALSE)
  }

  B <- as.integer(B)
  if (is.na(B) || B <= 1L) {
    stop("B must be an integer greater than 1.", call. = FALSE)
  }

  if (!is.null(seed)) {
    old_seed <- .Random.seed
    on.exit(assign(".Random.seed", old_seed, envir = .GlobalEnv), add = TRUE)
    set.seed(seed)
  }

  batch_size <- max(1L, as.integer(batch_size))
  total_batches <- ceiling(B / batch_size)

  coef_mat <- matrix(NA_real_, nrow = B, ncol = length(coef_names))
  colnames(coef_mat) <- coef_names
  pb <- NULL
  if (isTRUE(progress)) {
    pb <- utils::txtProgressBar(min = 0, max = B, style = 3)
    on.exit({
      close(pb)
    }, add = TRUE)
  }

  current_row <- 1L
  for (batch_idx in seq_len(total_batches)) {
    reps <- min(batch_size, B - current_row + 1L)
    batch_results <- vector("list", reps)
    for (j in seq_len(reps)) {
      idx <- sample.int(n, n, replace = TRUE)
      x_boot <- design_mat[idx, , drop = FALSE]
      y_boot <- response_num[idx]
      w_boot <- if (is.null(weights_num)) NULL else weights_num[idx]
      coef_boot <- if (fit_type == "lm") {
        .mlxs_bootstrap_case_lm_fit(x_boot, y_boot, w_boot)
      } else {
        .mlxs_bootstrap_case_glm_fit(
          x_boot,
          y_boot,
          w_boot,
          family = object$family,
          control = object$control,
          coef_start = object$coefficients
        )
      }
      if (length(coef_boot) != length(coef_names)) {
        stop("Coefficient count mismatch in bootstrap refit.", call. = FALSE)
      }
      names(coef_boot) <- coef_names
      batch_results[[j]] <- coef_boot
    }
    batch_mat <- do.call(rbind, batch_results)
    coef_mat[current_row:(current_row + reps - 1L), ] <- batch_mat
    current_row <- current_row + reps
    if (!is.null(pb)) {
      utils::setTxtProgressBar(pb, min(B, current_row - 1L))
    }
  }

  se <- apply(coef_mat, 2, stats::sd)
  list(se = se, samples = coef_mat, B = B, seed = seed)
}

.mlxs_bootstrap_case_lm_fit <- function(x_boot, y_boot, weights_boot) {
  x_mlx <- Rmlx::as_mlx(x_boot)
  y_mlx <- Rmlx::mlx_matrix(y_boot, ncol = 1)
  w_mlx <- if (is.null(weights_boot)) NULL else Rmlx::mlx_matrix(weights_boot, ncol = 1)
  fit <- mlxs_lm_fit(x_mlx, y_mlx, weights = w_mlx)
  .mlxs_as_numeric(fit$coefficients)
}

.mlxs_bootstrap_case_glm_fit <- function(x_boot,
                                         y_boot,
                                         weights_boot,
                                         family,
                                         control,
                                         coef_start) {
  fit <- .mlxs_glm_fit_core(
    design = x_boot,
    response = y_boot,
    weights_raw = weights_boot,
    family = family,
    control = control,
    coef_start = coef_start
  )
  .mlxs_as_numeric(fit$coefficients)
}

.mlxs_bootstrap_residual_lm <- function(object, B, seed, progress, batch_size) {
  qr_fit <- object$qr
  if (is.null(qr_fit)) {
    stop("QR decomposition not stored in mlxs_lm object.", call. = FALSE)
  }
  residuals <- .mlxs_as_numeric(object$residuals)
  fitted <- .mlxs_as_numeric(object$fitted.values)
  n <- length(residuals)
  if (length(fitted) != n) {
    stop("Mismatch between fitted values and residual lengths.", call. = FALSE)
  }
  centered_resid <- residuals - mean(residuals)
  coef_names <- object$coef_names
  if (is.null(coef_names)) {
    stop("Model must have named coefficients for bootstrap.", call. = FALSE)
  }
  if (!is.null(seed)) {
    old_seed <- .Random.seed
    on.exit(assign(".Random.seed", old_seed, envir = .GlobalEnv), add = TRUE)
    set.seed(seed)
  }
  batch_size <- max(1L, as.integer(batch_size))
  total_batches <- ceiling(B / batch_size)
  coef_mat <- matrix(NA_real_, nrow = B, ncol = length(coef_names))
  colnames(coef_mat) <- coef_names
  pb <- NULL
  if (isTRUE(progress)) {
    pb <- utils::txtProgressBar(min = 0, max = B, style = 3)
    on.exit({
      close(pb)
    }, add = TRUE)
  }
  current_row <- 1L
  Q_mlx <- qr_fit$Q
  R_mlx <- qr_fit$R
  for (batch_idx in seq_len(total_batches)) {
    reps <- min(batch_size, B - current_row + 1L)
    samples <- matrix(sample(centered_resid, n * reps, replace = TRUE), nrow = n, ncol = reps)
    y_batch <- matrix(fitted, nrow = n, ncol = reps) + samples
    y_batch_mlx <- Rmlx::as_mlx(y_batch)
    qty <- crossprod(Q_mlx, y_batch_mlx)
    coef_batch_mlx <- Rmlx::mlx_solve_triangular(R_mlx, qty, upper = TRUE)
    coef_batch_vec <- .mlxs_as_numeric(coef_batch_mlx)
    coef_batch <- matrix(coef_batch_vec, nrow = length(coef_names), ncol = reps)
    coef_mat[current_row:(current_row + reps - 1L), ] <- t(coef_batch)
    current_row <- current_row + reps
    if (!is.null(pb)) {
      utils::setTxtProgressBar(pb, min(B, current_row - 1L))
    }
  }
  se <- apply(coef_mat, 2, stats::sd)
  list(se = se, samples = coef_mat, B = B, seed = seed, method = "residual")
}

.mlxs_bootstrap_residual_glm <- function(object, B, seed, progress, batch_size) {
  qr_fit <- object$qr
  if (is.null(qr_fit)) {
    stop("QR decomposition not stored in mlxs_glm object.", call. = FALSE)
  }
  residuals <- .mlxs_as_numeric(object$residuals)
  fitted <- .mlxs_as_numeric(object$fitted.values)
  if (is.null(residuals) || is.null(fitted)) {
    stop("mlxs_glm object missing residuals or fitted values for bootstrap.", call. = FALSE)
  }
  n <- length(residuals)
  centered_resid <- residuals - mean(residuals)
  coef_names <- object$coef_names
  if (is.null(coef_names)) {
    stop("Model must have named coefficients for bootstrap.", call. = FALSE)
  }
  if (!is.null(seed)) {
    old_seed <- .Random.seed
    on.exit(assign(".Random.seed", old_seed, envir = .GlobalEnv), add = TRUE)
    set.seed(seed)
  }
  batch_size <- max(1L, as.integer(batch_size))
  total_batches <- ceiling(B / batch_size)
  coef_mat <- matrix(NA_real_, nrow = B, ncol = length(coef_names))
  colnames(coef_mat) <- coef_names
  pb <- NULL
  if (isTRUE(progress)) {
    pb <- utils::txtProgressBar(min = 0, max = B, style = 3)
    on.exit(close(pb), add = TRUE)
  }
  current_row <- 1L
  Q_mlx <- qr_fit$Q
  R_mlx <- qr_fit$R
  for (batch_idx in seq_len(total_batches)) {
    reps <- min(batch_size, B - current_row + 1L)
    samples <- matrix(sample(centered_resid, n * reps, replace = TRUE), nrow = n, ncol = reps)
    y_batch <- matrix(fitted, nrow = n, ncol = reps) + samples
    y_batch_mlx <- Rmlx::as_mlx(y_batch)
    qty <- crossprod(Q_mlx, y_batch_mlx)
    coef_batch_mlx <- Rmlx::mlx_solve_triangular(R_mlx, qty, upper = TRUE)
    coef_batch_vec <- .mlxs_as_numeric(coef_batch_mlx)
    coef_batch <- matrix(coef_batch_vec, nrow = length(coef_names), ncol = reps)
    coef_mat[current_row:(current_row + reps - 1L), ] <- t(coef_batch)
    current_row <- current_row + reps
    if (!is.null(pb)) {
      utils::setTxtProgressBar(pb, min(B, current_row - 1L))
    }
  }
  se <- apply(coef_mat, 2, stats::sd)
  list(se = se, samples = coef_mat, B = B, seed = seed, method = "residual")
}
NULL

#' @importFrom utils txtProgressBar setTxtProgressBar
#' @importFrom stats sd formula
