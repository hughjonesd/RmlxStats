.mlxs_glmnet_gaussian_chunk <- function(x_mlx,
                                        beta_mlx,
                                        eta_mlx,
                                        residual_mlx,
                                        y_mlx,
                                        n_obs,
                                        step,
                                        thresh,
                                        ridge_penalty,
                                        n_steps) {
  delta_max <- Rmlx::mlx_scalar(0)

  for (i in seq_len(n_steps)) {
    grad_mlx <- crossprod(x_mlx, residual_mlx) / n_obs
    if (ridge_penalty != 0) {
      grad_mlx <- grad_mlx + beta_mlx * ridge_penalty
    }

    beta_new_mlx <- .mlxs_soft_threshold(beta_mlx - step * grad_mlx, thresh)
    delta_mlx <- beta_new_mlx - beta_mlx

    eta_mlx <- eta_mlx + x_mlx %*% delta_mlx
    residual_mlx <- eta_mlx - y_mlx
    beta_mlx <- beta_new_mlx

    delta_max <- Rmlx::mlx_maximum(delta_max, max(abs(delta_mlx)))
  }

  list(
    beta = beta_mlx,
    eta = eta_mlx,
    residual = residual_mlx,
    delta_max = delta_max
  )
}

.mlxs_glmnet_gaussian_gram_chunk <- function(gram_mlx,
                                             xy_mlx,
                                             beta_mlx,
                                             z_mlx,
                                             t_prev,
                                             step,
                                             thresh,
                                             ridge_penalty,
                                             n_steps) {
  delta_max <- Rmlx::mlx_scalar(0)

  for (i in seq_len(n_steps)) {
    grad_mlx <- gram_mlx %*% z_mlx - xy_mlx
    if (ridge_penalty != 0) {
      grad_mlx <- grad_mlx + z_mlx * ridge_penalty
    }

    beta_new_mlx <- .mlxs_soft_threshold(z_mlx - step * grad_mlx, thresh)
    delta_mlx <- beta_new_mlx - beta_mlx
    delta_max <- Rmlx::mlx_maximum(delta_max, max(abs(delta_mlx)))

    t_next <- (1 + sqrt(1 + 4 * t_prev^2)) / 2
    z_mlx <- beta_new_mlx + ((t_prev - 1) / t_next) * delta_mlx
    beta_mlx <- beta_new_mlx
    t_prev <- t_next
  }

  list(
    beta = beta_mlx,
    z = z_mlx,
    t_prev = t_prev,
    delta_max = delta_max
  )
}

.mlxs_glmnet_binomial_chunk <- function(x_mlx,
                                        beta_mlx,
                                        intercept_mlx,
                                        eta_mlx,
                                        residual_mlx,
                                        y_mlx,
                                        ones_mlx,
                                        n_obs,
                                        step,
                                        thresh,
                                        ridge_penalty,
                                        n_steps,
                                        fit_intercept) {
  delta_max <- Rmlx::mlx_scalar(0)
  intercept_delta_max <- Rmlx::mlx_scalar(0)

  for (i in seq_len(n_steps)) {
    grad_mlx <- crossprod(x_mlx, residual_mlx) / n_obs
    if (ridge_penalty != 0) {
      grad_mlx <- grad_mlx + beta_mlx * ridge_penalty
    }

    beta_new_mlx <- .mlxs_soft_threshold(beta_mlx - step * grad_mlx, thresh)
    delta_mlx <- beta_new_mlx - beta_mlx

    if (fit_intercept) {
      intercept_delta_mlx <- step * (Rmlx::mlx_sum(residual_mlx) / n_obs)
      intercept_mlx <- intercept_mlx - intercept_delta_mlx
      eta_mlx <- eta_mlx + x_mlx %*% delta_mlx - ones_mlx * intercept_delta_mlx
      intercept_delta_max <- Rmlx::mlx_maximum(
        intercept_delta_max,
        abs(intercept_delta_mlx)
      )
    } else {
      eta_mlx <- eta_mlx + x_mlx %*% delta_mlx
    }

    mu_mlx <- 1 / (1 + exp(-eta_mlx))
    residual_mlx <- mu_mlx - y_mlx
    beta_mlx <- beta_new_mlx

    delta_max <- Rmlx::mlx_maximum(delta_max, max(abs(delta_mlx)))
  }

  list(
    beta = beta_mlx,
    intercept = intercept_mlx,
    eta = eta_mlx,
    residual = residual_mlx,
    delta_max = delta_max,
    intercept_delta_max = intercept_delta_max
  )
}
