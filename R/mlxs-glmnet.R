#' MLX-backed elastic net regression
#'
#' Fit lasso or elastic-net penalised regression paths using MLX tensors for
#' the heavy linear algebra. Currently supports Gaussian and binomial families
#' with an optional intercept and column standardisation.
#'
#' @note This function is early work. It is currently only faster than
#'   [glmnet::glmnet()] for very large data: roughly 100,000 observations
#'   and 1000 predictors.
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
#' @param use_strong_rules Should strong rules be used to screen out inactive
#'   predictors? This can speed up fitting for sparse problems.
#' @param block_size Number of coefficients to update per gradient evaluation.
#'   Set to 1 for classic coordinate descent; larger values (e.g., 16-64) batch
#'   updates for speed at the cost of a slightly more conservative step size.
#' @return An object of class `mlxs_glmnet` containing MLX arrays for the
#'   coefficient path (`beta`), intercepts (`a0`), and lambda sequence
#'   (`lambda`). Use `as.matrix()` / `as.numeric()` or the provided print
#'   method to materialise these on the host when needed.
#' @export
mlxs_glmnet <- function(
  x,
  y,
  family = mlxs_gaussian(),
  alpha = 1,
  lambda = NULL,
  nlambda = 100,
  lambda_min_ratio = 1e-4,
  standardize = TRUE,
  intercept = TRUE,
  maxit = 1000,
  tol = 1e-6,
  use_strong_rules = TRUE,
  block_size = 16L
) {
  family_name <- family$family
  if (!family_name %in% c("gaussian", "binomial", "quasibinomial")) {
    stop(
      "mlxs_glmnet() currently supports gaussian and binomial families.",
      call. = FALSE
    )
  }
  if (alpha <= 0) {
    stop(
      "alpha must be > 0 for the current MLX elastic-net implementation.",
      call. = FALSE
    )
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
    x_center <- Rmlx::mlx_reshape(
      attr(x_std_mlx, "scaled:center"),
      c(n_pred, 1)
    )
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
  z0_vals <- as.numeric(z0_mlx)
  lambda_max <- max(abs(z0_vals)) / max(alpha, 1e-8)
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
  grad_prev <- z0_mlx
  lambda_prev <- lambda[1]

  col_sq_sums_mlx <- Rmlx::colSums(x_std_mlx^2)
  col_sq_sums <- as.numeric(col_sq_sums_mlx)

  eta_mlx <- x_std_mlx %*% beta_mlx + ones_mlx * intercept_mlx
  mu_mlx <- family$linkinv(eta_mlx)
  residual_mlx <- mu_mlx - y_mlx

  # Coordinate descent path
  for (idx in seq_along(lambda)) {
    lambda_val <- lambda[idx]

    # Run coordinate descent for this lambda
    ridge_penalty <- lambda_val * (1 - alpha)
    l1_penalty <- lambda_val * alpha

    # Apply strong rules screening
    active_set <- rep(TRUE, n_pred)
    if (use_strong_rules && idx > 1) {
      grad_prev_host <- as.numeric(grad_prev)
      # Strong rule: discard j if |grad_j(lambda_prev)| < alpha * (2*lambda_k - lambda_{k-1})
      strong_threshold <- alpha * (2 * lambda_val - lambda_prev)
      active_set <- abs(grad_prev_host) >= strong_threshold

      # Handle NAs (treat as active to be safe)
      active_set[is.na(active_set)] <- TRUE

      # Always keep variables that were active at previous lambda
      beta_prev_active <- abs(as.numeric(beta_mlx)) > 0
      active_set <- active_set | beta_prev_active

      # Ensure at least one predictor is active
      if (!any(active_set, na.rm = TRUE)) {
        active_set[which.max(abs(grad_prev_host))] <- TRUE
      }
    }
    active_mask <- Rmlx::as_mlx(
      matrix(as.integer(active_set), ncol = 1),
      dtype = "bool"
    )

    # Fit with KKT checking loop
    repeat {
      active_set <- as.logical(active_mask)
      # Subset to active predictors (if all active, this is the full set)
      active_idx <- which(active_set)

      # Debug: check for empty active set
      if (length(active_idx) == 0) {
        stop(
          "Empty active set at lambda index ",
          idx,
          ". This should not happen - strong rules logic has a bug."
        )
      }

      x_active_mlx <- x_std_mlx[, active_idx, drop = FALSE]
      beta_active_mlx <- beta_mlx[active_idx, , drop = FALSE]
      col_sq_sums_active <- col_sq_sums[active_idx]

      # Define loss and gradient on active set
      if (family_name == "gaussian") {
        loss_active <- function(beta) {
          eta <- x_active_mlx %*% beta + ones_mlx * intercept_mlx
          loss_smooth <- Rmlx::mlx_mse_loss(eta, y_mlx, reduction = "mean")
          loss_smooth <- loss_smooth + 0.5 * ridge_penalty * sum(beta^2)
          loss_smooth
        }
        grad_active <- function(beta) {
          eta <- x_active_mlx %*% beta + ones_mlx * intercept_mlx
          residual <- y_mlx - eta
          grad <- -crossprod(x_active_mlx, residual) / n_obs
          grad <- grad + ridge_penalty * beta
          grad
        }
      } else {
        loss_active <- function(beta) {
          eta <- x_active_mlx %*% beta + ones_mlx * intercept_mlx
          mu <- 1 / (1 + exp(-eta))
          loss_smooth <- Rmlx::mlx_binary_cross_entropy(
            mu,
            y_mlx,
            reduction = "mean"
          )
          loss_smooth <- loss_smooth + 0.5 * ridge_penalty * sum(beta^2)
          loss_smooth
        }
        grad_active <- function(beta) {
          eta <- x_active_mlx %*% beta + ones_mlx * intercept_mlx
          mu <- 1 / (1 + exp(-eta))
          residual <- mu - y_mlx
          grad <- crossprod(x_active_mlx, residual) / n_obs
          grad <- grad + ridge_penalty * beta
          grad
        }
      }

      lipschitz_active <- col_sq_sums_active / n_obs + ridge_penalty
      if (family_name %in% c("binomial", "quasibinomial")) {
        lipschitz_active <- 0.25 * lipschitz_active
      }

      # Add small epsilon for numerical stability when ridge_penalty is 0
      lipschitz_active <- lipschitz_active + 1e-8

      lipschitz_active <- Rmlx::mlx_reshape(
        Rmlx::as_mlx(lipschitz_active),
        c(length(active_idx), 1)
      )

      block_size_active <- max(
        1L,
        min(as.integer(block_size), length(active_idx))
      )
      if (block_size_active > 1L && family_name == "gaussian") {
        block_indices <- split(
          seq_len(length(active_idx)),
          ceiling(seq_len(length(active_idx)) / block_size_active)
        )
        for (blk in block_indices) {
          if (length(blk) <= 1L) {
            next
          }
          x_block <- x_active_mlx[, blk, drop = FALSE]
          gram <- crossprod(x_block, x_block) / n_obs
          if (ridge_penalty > 0) {
            eye_blk <- Rmlx::mlx_eye(
              length(blk),
              dtype = gram$dtype,
              device = gram$device
            )
            gram <- gram + ridge_penalty * eye_blk
          }
          svd_vals <- Rmlx::svd(gram, nu = 0, nv = 0)
          spectral <- max(as.numeric(svd_vals$d))
          diag_vals <- as.numeric(lipschitz_active[blk, , drop = FALSE])
          diag_max <- max(diag_vals)
          if (spectral > diag_max && diag_max > 0) {
            scale <- spectral / diag_max
            lipschitz_active[blk, ] <- lipschitz_active[blk, ] * scale
          }
        }
      } else {
        block_size_active <- 1L
      }

      grad_cache <- NULL
      if (family_name == "gaussian") {
        grad_cache <- new.env(parent = emptyenv())
        grad_cache$type <- "gaussian"
        grad_cache$x <- x_active_mlx
        grad_cache$n_obs <- n_obs
        grad_cache$ridge_penalty <- ridge_penalty
        grad_cache$residual <- y_mlx -
          (x_active_mlx %*% beta_active_mlx + ones_mlx * intercept_mlx)
      } else if (family_name %in% c("binomial", "quasibinomial")) {
        grad_cache <- new.env(parent = emptyenv())
        grad_cache$type <- "binomial"
        grad_cache$x <- x_active_mlx
        grad_cache$n_obs <- n_obs
        grad_cache$ridge_penalty <- ridge_penalty
        grad_cache$y <- y_mlx
        grad_cache$eta <- x_active_mlx %*%
          beta_active_mlx +
          ones_mlx * intercept_mlx
        grad_cache$mu <- 1 / (1 + exp(-grad_cache$eta))
        grad_cache$residual <- grad_cache$mu - y_mlx
      }

      # Run coordinate descent (let it handle compilation internally)

      result <- Rmlx::mlx_coordinate_descent(
        loss_fn = loss_active,
        beta_init = beta_active_mlx,
        lambda = l1_penalty,
        grad_fn = grad_active,
        lipschitz = lipschitz_active,
        max_iter = maxit,
        tol = tol,
        block_size = block_size_active,
        grad_cache = grad_cache
      )

      # Check for NaN in result
      result_beta_vec <- as.numeric(result$beta)
      if (any(is.nan(result_beta_vec))) {
        warning(sprintf(
          "Coordinate descent produced NaN at lambda index %d (lambda=%.6f, converged=%s, iterations=%d)",
          idx,
          lambda_val,
          result$converged,
          result$iterations
        ))
      }

      # Expand beta back to full size (if all active, this is a no-op)
      beta_mlx <- Rmlx::mlx_zeros(c(n_pred, 1))
      beta_mlx[active_idx, ] <- result$beta

      # Update intercept
      if (intercept) {
        if (family_name == "gaussian") {
          residual_mlx <- y_mlx - x_std_mlx %*% beta_mlx
          intercept_mlx <- sum(residual_mlx) / n_obs
        } else if (family_name %in% c("binomial", "quasibinomial")) {
          for (intercept_iter in seq_len(2)) {
            eta <- x_std_mlx %*% beta_mlx + ones_mlx * intercept_mlx
            mu <- 1 / (1 + exp(-eta))
            residual <- mu - y_mlx
            w <- mu * (1 - mu)
            grad_intercept <- sum(residual) / n_obs
            hess_intercept <- sum(w) / n_obs
            intercept_new <- intercept_mlx -
              grad_intercept / (hess_intercept + 1e-8)
            if (as.logical(abs(intercept_new - intercept_mlx) < tol)) {
              intercept_mlx <- intercept_new
              break
            }
            intercept_mlx <- intercept_new
          }
        }
      } else {
        intercept_mlx <- Rmlx::as_mlx(0)
      }

      # Check KKT conditions if strong rules were used and some predictors were screened out
      if (!use_strong_rules || sum(active_set) == n_pred) {
        break # No screening, so no KKT check needed
      }

      # Compute gradient for all predictors
      eta_mlx <- x_std_mlx %*% beta_mlx + ones_mlx * intercept_mlx
      if (family_name == "gaussian") {
        residual_full <- y_mlx - eta_mlx
        grad_full <- -crossprod(x_std_mlx, residual_full) / n_obs
      } else {
        mu <- 1 / (1 + exp(-eta_mlx))
        residual_full <- mu - y_mlx
        grad_full <- crossprod(x_std_mlx, residual_full) / n_obs
      }
      grad_full <- grad_full + ridge_penalty * beta_mlx

      # Check KKT violations for inactive predictors
      inactive_mask <- !active_mask
      kkt_threshold <- l1_penalty + tol
      violations_mask <- inactive_mask & (abs(grad_full) > kkt_threshold)
      violations <- as.logical(violations_mask)

      # Handle NAs (treat as no violation to be safe)
      violations[is.na(violations)] <- FALSE

      if (!any(violations, na.rm = TRUE)) {
        break # No violations, we're done
      }

      # Add violators to active set and refit
      active_set <- active_set | violations
      active_mask <- active_mask | violations_mask
    }

    # Store results
    beta_store_mlx[, idx] <- beta_mlx
    intercept_store_mlx[idx, ] <- intercept_mlx

    # Force evaluation at each lambda (outer loop iteration)
    # See: https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html
    Rmlx::mlx_eval(beta_mlx)
    Rmlx::mlx_eval(intercept_mlx)

    # Update for strong rules (future optimization)
    eta_mlx <- x_std_mlx %*% beta_mlx + ones_mlx * intercept_mlx
    residual_mlx <- y_mlx - eta_mlx
    grad_prev <- crossprod(x_std_mlx, residual_mlx) / n_obs
    lambda_prev <- lambda_val
  }

  if (standardize) {
    beta_unscaled <- beta_store_mlx / x_scale
    intercept_adjustment <- Rmlx::colSums(beta_unscaled * x_center)
    intercept_adjustment <- Rmlx::mlx_reshape(
      intercept_adjustment,
      c(n_lambda, 1)
    )
    intercept_unscaled <- intercept_store_mlx - intercept_adjustment
  } else {
    beta_unscaled <- beta_store_mlx
    intercept_unscaled <- intercept_store_mlx
  }

  result <- list(
    a0 = intercept_unscaled,
    beta = beta_unscaled,
    lambda = Rmlx::as_mlx(lambda),
    lambda_numeric = lambda,
    alpha = alpha,
    family = family_name,
    standardize = standardize,
    intercept = intercept,
    x_center = x_center,
    x_scale = x_scale,
    coef_names = colnames(x),
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
