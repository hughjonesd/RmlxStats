#' MLX-backed elastic net regression
#'
#' Fit lasso or elastic-net penalised regression paths using MLX tensors for
#' the heavy linear algebra. Currently supports Gaussian and binomial families
#' with an optional intercept and column standardisation.
#'
#' @note This function is a proof-of-concept. On large dense problems it is
#'   typically several times slower than the highly optimised
#'   [glmnet::glmnet()] implementation.
#'
#' @param x Numeric matrix of predictors (observations in rows).
#' @param y Numeric response vector (for binomial, values must be 0/1).
#' @param family MLX-aware family object, e.g. [mlxs_gaussian()] or
#'   [mlxs_binomial()].
#' @param alpha Elastic-net mixing parameter (1 = lasso, currently alpha must
#'   be strictly positive).
#' @param lambda Optional decreasing sequence of penalty values. If `NULL`,
#'   a sequence of length `nlambda` is generated from `lambda_max` down to
#'   `lambda_max * lambda_min_ratio`.
#' @param nlambda Length of automatically generated lambda path.
#' @param lambda_min_ratio Smallest lambda as a fraction of `lambda_max`.
#' @param standardize Should columns of `x` be centred and scaled before
#'   fitting? Coefficients are returned on the original scale regardless.
#' @param intercept Should an intercept be fit?
#' @param maxit Maximum proximal-gradient iterations per lambda value.
#' @param tol Convergence tolerance on the coefficient updates.
#' @return An object of class `mlxs_glmnet` containing the fitted coefficient
#'   path, intercepts, lambda sequence, and scaling information.
#' @export
mlxs_glmnet <- function(x,
                        y,
                        family = mlxs_gaussian(),
                        alpha = 1,
                        lambda = NULL,
                        nlambda = 100,
                        lambda_min_ratio = 1e-4,
                        standardize = TRUE,
                        intercept = TRUE,
                        maxit = 1000,
                        tol = 1e-6) {
  family_name <- family$family
  if (!family_name %in% c("gaussian", "binomial", "quasibinomial")) {
    stop("mlxs_glmnet() currently supports gaussian and binomial families.", call. = FALSE)
  }
  if (alpha <= 0) {
    stop("alpha must be > 0 for the current MLX elastic-net implementation.", call. = FALSE)
  }

  x <- as.matrix(x)
  y <- as.numeric(y)
  if (nrow(x) != length(y)) {
    stop("x and y must have the same number of observations.", call. = FALSE)
  }

  n_obs <- nrow(x)
  n_pred <- ncol(x)

  # Convert to MLX early
  x_mlx <- Rmlx::as_mlx(x)
  y_mlx <- Rmlx::mlx_reshape(Rmlx::as_mlx(y), c(n_obs, 1))

  if (standardize) {
    x_std_mlx <- scale(x_mlx)
    x_center <- Rmlx::mlx_reshape(attr(x_std_mlx, "scaled:center"), c(n_pred, 1))
    x_scale <- Rmlx::mlx_reshape(attr(x_std_mlx, "scaled:scale"), c(n_pred, 1))
  } else {
    x_std_mlx <- x_mlx
    x_center <- Rmlx::mlx_zeros(c(n_pred, 1))
    x_scale <- Rmlx::mlx_ones(c(n_pred, 1))
  }

  if (family_name %in% c("binomial", "quasibinomial")) {
    if (!all(y %in% c(0, 1))) {
      stop("Binomial family requires a 0/1 response.", call. = FALSE)
    }
    p_hat <- mean(y)
    p_hat <- min(max(p_hat, 1e-6), 1 - 1e-6)
    intercept_val <- if (intercept) log(p_hat / (1 - p_hat)) else 0
    mu0 <- rep(p_hat, n_obs)
  } else {
    intercept_val <- if (intercept) mean(y) else 0
    mu0 <- rep(intercept_val, n_obs)
  }

  mu0_mlx <- Rmlx::as_mlx(matrix(mu0, ncol = 1))
  residual0_mlx <- mu0_mlx - y_mlx
  z0_mlx <- crossprod(x_std_mlx, residual0_mlx) / n_obs
  z0 <- as.numeric(z0_mlx)
  lambda_max <- max(abs(z0)) / max(alpha, 1e-8)
  if (is.finite(lambda_max) && lambda_max == 0) {
    lambda_max <- 1
  }

  if (is.null(lambda)) {
    lambda_min <- lambda_max * lambda_min_ratio
    lambda <- exp(seq(log(lambda_max), log(lambda_min), length.out = nlambda))
  } else {
    lambda <- sort(as.numeric(lambda), decreasing = TRUE)
  }
  n_lambda <- length(lambda)

  ones_mlx <- Rmlx::as_mlx(matrix(1, nrow = n_obs, ncol = 1))
  beta_mlx <- Rmlx::mlx_zeros(c(n_pred, 1))
  intercept_mlx <- Rmlx::as_mlx(matrix(intercept_val, nrow = 1, ncol = 1))

  # Keep stores as MLX to avoid per-lambda conversions
  beta_store_mlx <- Rmlx::mlx_zeros(c(n_pred, n_lambda))
  intercept_store_mlx <- Rmlx::mlx_zeros(c(n_lambda, 1))
  grad_prev <- z0
  lambda_prev <- lambda[1]

  col_sq_sums_mlx <- Rmlx::colSums(x_std_mlx^2)
  col_sq_sums <- as.numeric(col_sq_sums_mlx)
  base_lipschitz <- max(col_sq_sums) / n_obs
  if (family_name %in% c("binomial", "quasibinomial")) {
    base_lipschitz <- 0.25 * base_lipschitz
  }

  # Tolerance as MLX scalar for comparisons
  tol_mlx <- Rmlx::as_mlx(matrix(tol, nrow = 1, ncol = 1))

  # Get compiled iteration function (caches on first call)
  iter_func <- .get_compiled_iteration()

  # Family flag for compiled function (1 = gaussian, 0 = binomial)
  is_gaussian_flag <- Rmlx::as_mlx(matrix(
    if (family_name == "gaussian") 1 else 0,
    nrow = 1, ncol = 1
  ))

  eta_mlx <- x_std_mlx %*% beta_mlx + ones_mlx * intercept_mlx
  mu_mlx <- family$linkinv(eta_mlx)
  residual_mlx <- mu_mlx - y_mlx

  for (idx in seq_along(lambda)) {
    lambda_val <- lambda[idx]
    step <- 1 / (base_lipschitz + lambda_val * (1 - alpha))
    if (!is.finite(step) || step <= 0) {
      step <- 1e-3
    }

    if (idx == 1) {
      active_idx <- seq_len(n_pred)
    } else {
      cutoff <- alpha * (2 * lambda_val - lambda_prev)
      strong_set <- which(abs(grad_prev) > cutoff)
      # Convert previous beta column to R only for strong rules check
      beta_numeric <- as.numeric(beta_store_mlx[, idx - 1])
      nonzero_set <- which(beta_numeric != 0)
      active_idx <- sort(unique(c(strong_set, nonzero_set)))
      if (length(active_idx) == 0) {
        active_idx <- which.max(abs(grad_prev))
      }
    }

    # Precompute threshold for compiled function
    thresh <- lambda_val * alpha * step

    for (iter in seq_len(maxit)) {
      x_active <- x_std_mlx[, active_idx, drop = FALSE]
      beta_prev_subset <- beta_mlx[active_idx, , drop = FALSE]

      # Call compiled iteration function
      result <- iter_func(
        x_active, beta_prev_subset, residual_mlx,
        intercept_mlx, eta_mlx, y_mlx, ones_mlx,
        n_obs, step, lambda_val, alpha, thresh,
        is_gaussian_flag
      )

      # Extract results
      beta_mlx[active_idx, ] <- result$beta_new
      intercept_mlx <- result$intercept_new
      eta_mlx <- result$eta_new
      residual_mlx <- result$residual_new

      # Convergence check using MLX operations
      delta_beta_max <- max(abs(result$delta_beta))
      intercept_delta_abs <- abs(result$intercept_delta)
      if (as.logical(delta_beta_max < tol) && as.logical(intercept_delta_abs < tol)) {
        break
      }
    }

    # Store in MLX - no conversion until the end
    beta_store_mlx[, idx] <- beta_mlx
    intercept_store_mlx[idx, ] <- intercept_mlx

    grad_prev <- as.numeric(crossprod(x_std_mlx, residual_mlx) / n_obs)
    lambda_prev <- lambda_val
  }

  if (standardize) {
    beta_unscaled <- beta_store_mlx / x_scale
    intercept_adjustment <- Rmlx::colSums(beta_unscaled * x_center)
    intercept_adjustment <- Rmlx::mlx_reshape(intercept_adjustment, c(n_lambda, 1))
    intercept_unscaled <- intercept_store_mlx - intercept_adjustment
  } else {
    beta_unscaled <- beta_store_mlx
    intercept_unscaled <- intercept_store_mlx
  }
  
  # Convert to R only once at the very end
  beta_unscaled <- as.matrix(beta_unscaled)
  intercept_unscaled <- as.numeric(intercept_unscaled)

  rownames(beta_unscaled) <- colnames(x)
  result <- list(
    a0 = intercept_unscaled,
    beta = beta_unscaled,
    lambda = lambda,
    alpha = alpha,
    family = family_name,
    standardize = standardize,
    intercept = intercept,
    x_center = x_center,
    x_scale = x_scale,
    call = match.call()
  )

  class(result) <- "mlxs_glmnet"
  result
}

.mlxs_soft_threshold <- function(z, thresh) {
  # Soft threshold: sign(z) * max(|z| - thresh, 0)
  # Simplified to reduce temporary allocations
  abs_z <- abs(z)
  magnitude <- Rmlx::mlx_maximum(abs_z - thresh, 0)
  # Compute sign as z / |z|, with small epsilon to avoid division by zero
  sign_z <- z / (abs_z + 1e-10)
  magnitude * sign_z
}
