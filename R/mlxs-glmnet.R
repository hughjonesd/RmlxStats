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

  if (standardize) {
    x_center <- colMeans(x)
    x_scale <- apply(x, 2, stats::sd)
    x_scale[is.na(x_scale) | x_scale == 0] <- 1
    x_mlx_scaled <- scale(Rmlx::as_mlx(x), center = x_center, scale = x_scale)
    x_std <- as.matrix(x_mlx_scaled)
    attr(x_std, "scaled:center") <- x_center
    attr(x_std, "scaled:scale") <- x_scale
  } else {
    x_std <- x
    x_center <- rep(0, n_pred)
    x_scale <- rep(1, n_pred)
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

  residual0 <- mu0 - y
  z0 <- drop(crossprod(x_std, residual0) / n_obs)
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
  x_mlx <- Rmlx::as_mlx(x_std)
  y_mlx <- Rmlx::as_mlx(matrix(y, ncol = 1))
  beta_mlx <- Rmlx::as_mlx(matrix(0, nrow = n_pred, ncol = 1))

  beta_store <- matrix(0, nrow = n_pred, ncol = n_lambda)
  intercept_store <- numeric(n_lambda)
  grad_prev <- z0
  lambda_prev <- lambda[1]

  col_sq_sums <- colSums(x_std^2)
  base_lipschitz <- max(col_sq_sums) / n_obs
  if (family_name %in% c("binomial", "quasibinomial")) {
    base_lipschitz <- 0.25 * base_lipschitz
  }

  eta_mlx <- x_mlx %*% beta_mlx + ones_mlx * Rmlx::as_mlx(intercept_val)
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
      beta_numeric <- beta_store[, idx - 1]
      nonzero_set <- which(beta_numeric != 0)
      active_idx <- sort(unique(c(strong_set, nonzero_set)))
      if (length(active_idx) == 0) {
        active_idx <- which.max(abs(grad_prev))
      }
    }

    for (iter in seq_len(maxit)) {
      x_active <- x_mlx[, active_idx, drop = FALSE]
      beta_prev_subset <- beta_mlx[active_idx, , drop = FALSE]

      grad_active <- crossprod(x_active, residual_mlx) / n_obs
      if (alpha < 1) {
        grad_active <- grad_active + beta_prev_subset * (lambda_val * (1 - alpha))
      }

      beta_temp <- beta_prev_subset - grad_active * step
      thresh <- lambda_val * alpha * step
      beta_new_subset <- .mlxs_soft_threshold(beta_temp, thresh)
      delta_beta <- beta_new_subset - beta_prev_subset

      beta_mlx[active_idx, ] <- beta_new_subset

      residual_sum <- Rmlx::mlx_sum(residual_mlx)
      intercept_grad <- residual_sum / n_obs
      intercept_delta <- step * .mlxs_as_numeric(intercept_grad)
      intercept_val <- intercept_val - intercept_delta

      delta_exceeds <- Rmlx::mlx_any(abs(delta_beta) > tol)
      if (isTRUE(.mlxs_as_numeric(delta_exceeds))) {
        eta_mlx <- eta_mlx + x_active %*% delta_beta
      }
      if (abs(intercept_delta) > tol) {
        eta_mlx <- eta_mlx - ones_mlx * intercept_delta
      }

      mu_mlx <- family$linkinv(eta_mlx)
      residual_mlx <- mu_mlx - y_mlx

      max_change <- max(abs(.mlxs_as_numeric(delta_beta)))
      if (max_change < tol && abs(intercept_delta) < tol) {
        break
      }
    }

    beta_store[, idx] <- .mlxs_as_numeric(beta_mlx)
    intercept_store[idx] <- intercept_val

    grad_prev <- .mlxs_as_numeric(crossprod(x_mlx, residual_mlx) / n_obs)
    lambda_prev <- lambda_val
  }

  if (standardize) {
    beta_unscaled <- sweep(beta_store, 1, x_scale, "/")
    intercept_unscaled <- intercept_store - colSums(beta_unscaled * x_center)
  } else {
    beta_unscaled <- beta_store
    intercept_unscaled <- intercept_store
  }

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
  thresh_vec <- Rmlx::as_mlx(thresh)
  zero_like <- Rmlx::mlx_zeros_like(z)
  one_like <- Rmlx::mlx_full(dim(z), 1)
  neg_one_like <- Rmlx::mlx_full(dim(z), -1)
  abs_z <- abs(z)
  magnitude <- abs_z - thresh_vec
  magnitude <- Rmlx::mlx_where(magnitude > zero_like, magnitude, zero_like)
  sign_vec <- Rmlx::mlx_where(z >= zero_like, one_like, neg_one_like)
  magnitude * sign_vec
}
