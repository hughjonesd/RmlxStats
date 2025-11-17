# Compiled inner loop for mlxs_glmnet
# Separated out to enable mlx_compile() optimization

.mlxs_glmnet_one_iteration <- function(
  x_active,
  beta_prev_subset,
  residual_mlx,
  intercept_prev,
  eta_mlx,
  y_mlx,
  ones_mlx,
  n_obs,
  step,
  lambda_val,
  alpha,
  thresh,
  is_gaussian_flag
) {
  # Gradient computation
  grad_active <- crossprod(x_active, residual_mlx) / n_obs

  # Ridge penalty (always compute, will be zero if alpha=1)
  grad_active <- grad_active + beta_prev_subset * (lambda_val * (1 - alpha))

  # Proximal gradient step
  beta_temp <- beta_prev_subset - grad_active * step

  # Soft threshold
  abs_beta <- abs(beta_temp)
  magnitude <- Rmlx::mlx_maximum(abs_beta - thresh, 0)
  sign_beta <- beta_temp / (abs_beta + 1e-10)
  beta_new_subset <- magnitude * sign_beta

  delta_beta <- beta_new_subset - beta_prev_subset

  # Intercept update
  residual_sum <- Rmlx::mlx_sum(residual_mlx)
  intercept_grad <- residual_sum / n_obs
  intercept_delta <- intercept_grad * step
  intercept_new <- intercept_prev - intercept_delta

  # Update eta
  eta_new <- eta_mlx + x_active %*% delta_beta - ones_mlx * intercept_delta

  # Link inverse (gaussian: identity, binomial: logistic)
  # is_gaussian_flag is 1 for gaussian, 0 for binomial
  mu_binomial <- 1 / (1 + exp(-eta_new))
  mu <- Rmlx::mlx_where(is_gaussian_flag > 0.5, eta_new, mu_binomial)

  residual_new <- mu - y_mlx

  # Return list with named elements (Rmlx issue #16 now fixed)
  list(
    beta_new = beta_new_subset,
    delta_beta = delta_beta,
    intercept_new = intercept_new,
    intercept_delta = intercept_delta,
    eta_new = eta_new,
    residual_new = residual_new
  )
}

# Environment to cache compiled version (avoids locked binding issues)
.mlxs_glmnet_cache <- new.env(parent = emptyenv())
.mlxs_glmnet_cache$compiled <- NULL

# Initialize compiled version (called on first use)
.get_compiled_iteration <- function() {
  if (is.null(.mlxs_glmnet_cache$compiled)) {
    # Try to compile, fall back to uncompiled if it fails
    .mlxs_glmnet_cache$compiled <- tryCatch(
      {
        Rmlx::mlx_compile(.mlxs_glmnet_one_iteration)
      },
      error = function(e) {
        warning(
          "Could not compile mlxs_glmnet inner loop: ",
          conditionMessage(e),
          "\nUsing uncompiled version (slower).",
          call. = FALSE
        )
        .mlxs_glmnet_one_iteration
      }
    )
  }
  .mlxs_glmnet_cache$compiled
}
