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

  eta_mlx <- x_std_mlx %*% beta_mlx + ones_mlx * intercept_mlx
  mu_mlx <- family$linkinv(eta_mlx)
  residual_mlx <- mu_mlx - y_mlx

  # Coordinate descent path
  for (idx in seq_along(lambda)) {
    lambda_val <- lambda[idx]

    # Run coordinate descent for this lambda
    ridge_penalty <- lambda_val * (1 - alpha)
    l1_penalty <- lambda_val * alpha

    # Define loss and gradient based on family
    if (family_name == "gaussian") {
      loss_fn <- function(beta) {
        eta <- x_std_mlx %*% beta + ones_mlx * intercept_mlx
        loss_smooth <- Rmlx::mlx_mse_loss(eta, y_mlx, reduction = "mean")
        if (ridge_penalty > 0) {
          loss_smooth <- loss_smooth + 0.5 * ridge_penalty * sum(beta^2)
        }
        loss_smooth
      }

      grad_fn <- function(beta) {
        eta <- x_std_mlx %*% beta + ones_mlx * intercept_mlx
        residual <- y_mlx - eta
        grad <- -crossprod(x_std_mlx, residual) / n_obs
        if (ridge_penalty > 0) {
          grad <- grad + ridge_penalty * beta
        }
        grad
      }

      lipschitz <- col_sq_sums / n_obs + ridge_penalty

    } else if (family_name %in% c("binomial", "quasibinomial")) {
      loss_fn <- function(beta) {
        eta <- x_std_mlx %*% beta + ones_mlx * intercept_mlx
        mu <- 1 / (1 + exp(-eta))
        loss_smooth <- Rmlx::mlx_binary_cross_entropy(mu, y_mlx, reduction = "mean")
        if (ridge_penalty > 0) {
          loss_smooth <- loss_smooth + 0.5 * ridge_penalty * sum(beta^2)
        }
        loss_smooth
      }

      grad_fn <- function(beta) {
        eta <- x_std_mlx %*% beta + ones_mlx * intercept_mlx
        mu <- 1 / (1 + exp(-eta))
        residual <- mu - y_mlx
        grad <- crossprod(x_std_mlx, residual) / n_obs
        if (ridge_penalty > 0) {
          grad <- grad + ridge_penalty * beta
        }
        grad
      }

      # For binomial, Lipschitz constant scaled by max(mu * (1-mu)) = 0.25
      lipschitz <- 0.25 * col_sq_sums / n_obs + ridge_penalty

    } else {
      stop("Unsupported family: ", family_name, call. = FALSE)
    }

    # Run coordinate descent for beta with current intercept fixed
    result <- Rmlx::mlx_coordinate_descent(
      loss_fn = loss_fn,
      beta_init = beta_mlx,
      lambda = l1_penalty,
      grad_fn = grad_fn,
      lipschitz = lipschitz,
      compile = FALSE,
      max_iter = maxit,
      tol = tol
    )

    beta_mlx <- result$beta

    # Update intercept analytically after beta convergence
    if (intercept) {
      if (family_name == "gaussian") {
        # For Gaussian, intercept is mean of residuals
        residual_mlx <- y_mlx - x_std_mlx %*% beta_mlx
        intercept_mlx <- sum(residual_mlx) / n_obs
      } else if (family_name %in% c("binomial", "quasibinomial")) {
        # For binomial, use Newton-Raphson for intercept (1-2 iterations)
        for (intercept_iter in seq_len(2)) {
          eta <- x_std_mlx %*% beta_mlx + ones_mlx * intercept_mlx
          mu <- 1 / (1 + exp(-eta))
          residual <- mu - y_mlx
          w <- mu * (1 - mu)
          grad_intercept <- sum(residual) / n_obs
          hess_intercept <- sum(w) / n_obs
          intercept_new <- intercept_mlx - grad_intercept / (hess_intercept + 1e-8)

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
