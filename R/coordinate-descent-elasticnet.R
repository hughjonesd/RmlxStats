#' Elastic Net using General Coordinate Descent
#'
#' Solves: min_beta 0.5/n * ||y - X*beta||^2 + lambda * (alpha * ||beta||_1 + 0.5 * (1-alpha) * ||beta||^2)
#'
#' @param X Design matrix (n x p MLX tensor)
#' @param y Response (n x 1 MLX tensor)
#' @param lambda Penalty parameter
#' @param alpha Elastic net mixing (1 = lasso, 0 = ridge)
#' @param beta_init Initial beta
#' @param batch_size Coordinates per batch
#' @param compile Whether to compile
#' @param max_iter Maximum iterations
#' @param tol Convergence tolerance
#' @return List with beta, n_iter, converged
#' @export
coordinate_descent_elasticnet <- function(X, y, lambda, alpha,
                                          beta_init,
                                          batch_size = NULL,
                                          compile = FALSE,
                                          max_iter = 1000,
                                          tol = 1e-6) {
  n_obs <- nrow(X)
  n_pred <- ncol(X)

  # For elastic net, the smooth part is:
  # f(beta) = 0.5/n * ||y - X*beta||^2 + 0.5 * lambda * (1-alpha) * ||beta||^2
  # The L1 part is: lambda * alpha * ||beta||_1

  # Precompute column norms for Lipschitz constants
  col_sq_sums <- as.numeric(Rmlx::colSums(X^2))
  ridge_penalty <- lambda * (1 - alpha)
  lipschitz <- col_sq_sums / n_obs + ridge_penalty

  # Loss function (smooth part only)
  loss_fn <- function(beta) {
    residual <- y - X %*% beta
    loss_smooth <- sum(residual^2) / (2 * n_obs)
    if (ridge_penalty > 0) {
      loss_smooth <- loss_smooth + 0.5 * ridge_penalty * sum(beta^2)
    }
    loss_smooth
  }

  # Gradient function (can compute analytically for efficiency)
  grad_fn <- function(beta) {
    residual <- y - X %*% beta
    grad <- -crossprod(X, residual) / n_obs
    if (ridge_penalty > 0) {
      grad <- grad + ridge_penalty * beta
    }
    grad
  }

  # Call general coordinate descent
  result <- coordinate_descent(
    loss_fn = loss_fn,
    beta_init = beta_init,
    lambda = lambda * alpha,  # L1 penalty only
    grad_fn = grad_fn,
    lipschitz = lipschitz,
    batch_size = batch_size,
    compile = compile,
    max_iter = max_iter,
    tol = tol
  )

  result
}
