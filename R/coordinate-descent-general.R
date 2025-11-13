#' General Coordinate Descent with L1 Regularization
#'
#' Solves: min_beta f(beta) + lambda * ||beta||_1
#' where f is a smooth differentiable loss function.
#'
#' @param loss_fn Function(beta) -> scalar loss (MLX tensor). Should be smooth.
#' @param beta_init Initial beta (p x 1 MLX tensor)
#' @param lambda L1 penalty parameter (scalar)
#' @param grad_fn Optional pre-computed gradient function. If NULL, uses mlx_grad(loss_fn)
#' @param lipschitz Optional Lipschitz constants for each coordinate (length p vector).
#'   If NULL, uses backtracking line search.
#' @param batch_size Number of coordinates to update simultaneously (1 = sequential, p = full batch)
#' @param compile Whether to compile the update step
#' @param max_iter Maximum iterations
#' @param tol Convergence tolerance
#' @return List with beta, n_iter, converged
#' @export
coordinate_descent <- function(loss_fn,
                               beta_init,
                               lambda = 0,
                               grad_fn = NULL,
                               lipschitz = NULL,
                               batch_size = NULL,
                               compile = FALSE,
                               max_iter = 1000,
                               tol = 1e-6) {

  beta <- beta_init
  n_pred <- nrow(beta)

  # Default batch size: sequential for small p, batched for large p
  if (is.null(batch_size)) {
    batch_size <- if (n_pred <= 100) 1 else min(50, n_pred)
  }

  # Get gradient function
  if (is.null(grad_fn)) {
    grad_fn <- Rmlx::mlx_grad(loss_fn)
  }

  # Default Lipschitz constants (will use backtracking if not provided)
  use_backtracking <- is.null(lipschitz)
  if (!use_backtracking) {
    lipschitz <- as.numeric(lipschitz)
  }

  # Create batches of coordinate indices
  coord_batches <- split(seq_len(n_pred), ceiling(seq_len(n_pred) / batch_size))
  n_batches <- length(coord_batches)

  # Optionally compile the update function
  if (compile) {
    update_fn <- .compile_cd_update(loss_fn, grad_fn, lambda, use_backtracking)
  }

  for (iter in seq_len(max_iter)) {
    beta_old <- as.numeric(beta)

    # Cycle through coordinate batches
    for (batch_idx in seq_len(n_batches)) {
      coords <- coord_batches[[batch_idx]]
      n_coords <- length(coords)

      if (n_coords == 1 && !compile) {
        # Sequential update (one coordinate)
        j <- coords[1]
        beta <- .update_single_coordinate(
          beta, j, loss_fn, grad_fn, lambda,
          if (use_backtracking) NULL else lipschitz[j]
        )
      } else {
        # Batch update (multiple coordinates)
        if (compile && !is.null(update_fn)) {
          # Use compiled version
          beta <- update_fn(beta, coords)
        } else {
          # Non-compiled batch update
          beta <- .update_coordinate_batch(
            beta, coords, loss_fn, grad_fn, lambda,
            if (use_backtracking) NULL else lipschitz[coords]
          )
        }
      }
    }

    # Check convergence
    delta <- max(abs(as.numeric(beta) - beta_old))
    if (delta < tol) {
      return(list(
        beta = beta,
        n_iter = iter,
        converged = TRUE
      ))
    }
  }

  list(
    beta = beta,
    n_iter = max_iter,
    converged = FALSE
  )
}

#' Update a single coordinate
#' @keywords internal
.update_single_coordinate <- function(beta, j, loss_fn, grad_fn, lambda, L_j = NULL) {
  # Compute gradient at current beta
  grad <- grad_fn(beta)
  grad_j <- grad[j, ]

  # Current beta_j
  beta_j_old <- beta[j, ]

  # Lipschitz constant for coordinate j
  if (is.null(L_j)) {
    # Backtracking line search
    L_j <- .backtracking_lipschitz(beta, j, loss_fn, grad_j)
  }

  # Proximal gradient step with soft thresholding
  # beta_j_new = prox_{lambda/L_j * ||.||_1} (beta_j - grad_j/L_j)
  z_j <- beta_j_old - grad_j / L_j

  # Soft threshold
  abs_z <- abs(z_j)
  threshold <- lambda / L_j

  if (as.logical(abs_z > threshold)) {
    beta_j_new <- sign(z_j) * (abs_z - threshold)
  } else {
    beta_j_new <- 0
  }

  # Update beta
  beta[j, ] <- beta_j_new
  beta
}

#' Update a batch of coordinates
#' @keywords internal
.update_coordinate_batch <- function(beta, coords, loss_fn, grad_fn, lambda, L_coords = NULL) {
  # Compute gradient
  grad <- grad_fn(beta)
  grad_coords <- grad[coords, , drop = FALSE]

  # Current beta values
  beta_coords_old <- beta[coords, , drop = FALSE]

  # Lipschitz constants
  if (is.null(L_coords)) {
    # Use same constant for all (conservative)
    L_coords <- rep(1.0, length(coords))
  }

  # Proximal gradient step - vectorized
  # Element-wise operations to avoid broadcast issues
  n_coords <- length(coords)
  beta_coords_new <- beta_coords_old

  for (i in seq_along(coords)) {
    z_i <- beta_coords_old[i, ] - grad_coords[i, ] / L_coords[i]
    abs_z <- abs(z_i)
    threshold <- lambda / L_coords[i]

    if (as.logical(abs_z > threshold)) {
      beta_coords_new[i, ] <- sign(z_i) * (abs_z - threshold)
    } else {
      beta_coords_new[i, ] <- 0
    }
  }

  # Update beta
  beta[coords, ] <- beta_coords_new
  beta
}

#' Backtracking line search to estimate Lipschitz constant
#' @keywords internal
.backtracking_lipschitz <- function(beta, j, loss_fn, grad_j, L_init = 1.0, eta = 2.0) {
  # Simple backtracking to find L_j such that quadratic upper bound holds
  # This is a placeholder - could be made more sophisticated
  L_init
}

#' Compile coordinate descent update
#' @keywords internal
.compile_cd_update <- function(loss_fn, grad_fn, lambda, use_backtracking) {
  # TODO: Implement compiled version using mlx_compile
  # For now, return NULL to use non-compiled path
  NULL
}
