#' Coordinate Descent for Elastic Net
#'
#' Solves: min_beta 0.5 * ||y - X*beta - intercept||^2 + lambda * (alpha * ||beta||_1 + 0.5 * (1-alpha) * ||beta||^2)
#'
#' @param X Design matrix (n x p), as MLX tensor
#' @param y Response vector (n x 1), as MLX tensor
#' @param lambda Penalty parameter (scalar)
#' @param alpha Elastic net mixing (1 = lasso, 0 = ridge)
#' @param beta_init Initial beta (p x 1), as MLX tensor
#' @param intercept_init Initial intercept (scalar)
#' @param fit_intercept Whether to fit intercept
#' @param max_iter Maximum iterations
#' @param tol Convergence tolerance
#' @param col_sq_sums Pre-computed column norms (||X_j||^2 for each j)
#' @return List with beta, intercept, n_iter
.coordinate_descent_elastic_net <- function(X, y, lambda, alpha,
                                             beta_init, intercept_init,
                                             fit_intercept = TRUE,
                                             max_iter = 1000,
                                             tol = 1e-6,
                                             col_sq_sums = NULL) {
  n_obs <- nrow(X)
  n_pred <- ncol(X)

  if (is.null(col_sq_sums)) {
    col_sq_sums <- as.numeric(Rmlx::colSums(X^2))
  }

  beta <- beta_init
  intercept <- intercept_init
  penalty <- lambda * alpha
  ridge_penalty <- lambda * (1 - alpha)

  # Maintain residual
  ones_mlx <- Rmlx::as_mlx(matrix(1, nrow = n_obs, ncol = 1))
  residual <- y - X %*% beta - ones_mlx * intercept

  for (iter in seq_len(max_iter)) {
    beta_old <- as.numeric(beta)
    intercept_old <- as.numeric(intercept)

    # Update intercept
    if (fit_intercept) {
      intercept_new <- sum(residual) / n_obs
      residual <- residual + ones_mlx * (intercept - intercept_new)
      intercept <- intercept_new
    }

    # Update each coordinate
    for (j in seq_len(n_pred)) {
      x_j <- X[, j, drop = FALSE]
      beta_j_old <- beta[j, ]

      # Compute update
      grad_j <- sum(x_j * residual) / n_obs
      z_j <- grad_j + beta_j_old * (col_sq_sums[j] / n_obs)
      denom_j <- col_sq_sums[j] / n_obs + ridge_penalty

      # Soft threshold
      abs_z <- abs(z_j)
      if (as.logical(abs_z > penalty)) {
        beta_j_new <- sign(z_j) * (abs_z - penalty) / denom_j
      } else {
        beta_j_new <- 0
      }

      # Update residual
      beta_change <- beta_j_new - beta_j_old
      if (as.logical(abs(beta_change) > 1e-10)) {
        residual <- residual - x_j * beta_change
      }

      beta[j, ] <- beta_j_new
    }

    # Check convergence
    delta_beta <- max(abs(as.numeric(beta) - beta_old))
    delta_intercept <- abs(as.numeric(intercept) - intercept_old)
    if (delta_beta < tol && delta_intercept < tol) {
      break
    }
  }

  list(
    beta = beta,
    intercept = intercept,
    n_iter = iter
  )
}
