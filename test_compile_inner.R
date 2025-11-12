#!/usr/bin/env Rscript
# Test if we can compile the inner loop

library(RmlxStats)
library(Rmlx)

# Create a compiled version of one iteration
one_iteration <- function(x_active, beta_prev_subset, residual_mlx,
                         intercept_prev, eta_mlx, y_mlx, ones_mlx,
                         n_obs, step, lambda_val, alpha,
                         is_gaussian) {

  # Gradient
  grad_active <- crossprod(x_active, residual_mlx) / n_obs

  # Ridge penalty - use mlx_where instead of if
  ridge_term <- beta_prev_subset * (lambda_val * (1 - alpha))
  # If alpha == 1, ridge_term should be 0, but multiplying by (1-alpha) already does this
  grad_active <- grad_active + ridge_term

  # Proximal gradient step
  beta_temp <- beta_prev_subset - grad_active * step

  # Soft threshold (inline it)
  thresh <- lambda_val * alpha * step
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
  eta_new <- eta_mlx + x_active %*% delta_beta
  eta_new <- eta_new - ones_mlx * intercept_delta

  # Link inverse - use mlx_where based on is_gaussian flag
  # For gaussian: mu = eta
  # For binomial: mu = 1/(1 + exp(-eta))
  mu_gaussian <- eta_new
  mu_binomial <- 1 / (1 + exp(-eta_new))
  mu <- Rmlx::mlx_where(is_gaussian > 0, mu_gaussian, mu_binomial)

  residual_new <- mu - y_mlx

  # Return updated values
  list(
    beta_new = beta_new_subset,
    delta_beta = delta_beta,
    intercept_new = intercept_new,
    intercept_delta = intercept_delta,
    eta_new = eta_new,
    residual_new = residual_new
  )
}

cat("Testing uncompiled version...\n")
set.seed(42)
n <- 100
p <- 20
x <- matrix(rnorm(n*p), n, p)
y <- rnorm(n)

x_mlx <- Rmlx::as_mlx(x)
y_mlx <- Rmlx::as_mlx(matrix(y, ncol=1))
beta_mlx <- Rmlx::mlx_zeros(c(p, 1))
intercept_mlx <- Rmlx::as_mlx(matrix(0, 1, 1))
ones_mlx <- Rmlx::as_mlx(matrix(1, n, 1))
eta_mlx <- x_mlx %*% beta_mlx + ones_mlx * intercept_mlx
residual_mlx <- eta_mlx - y_mlx
is_gaussian <- Rmlx::as_mlx(matrix(1, 1, 1))

# Time uncompiled
system.time({
  for (i in 1:100) {
    result <- one_iteration(
      x_mlx, beta_mlx, residual_mlx, intercept_mlx,
      eta_mlx, y_mlx, ones_mlx, n, 0.01, 0.1, 1, is_gaussian
    )
    mlx_eval(result$residual_new)
  }
})

cat("\nAttempting compilation...\n")
tryCatch({
  one_iteration_compiled <- mlx_compile(one_iteration)
  cat("Compilation succeeded!\n")

  cat("\nTesting compiled version...\n")
  result <- one_iteration_compiled(
    x_mlx, beta_mlx, residual_mlx, intercept_mlx,
    eta_mlx, y_mlx, ones_mlx, n, 0.01, 0.1, 1, is_gaussian
  )

  cat("Result class:", class(result), "\n")
  cat("Result names:", names(result), "\n")
  if (!is.null(result$residual_new)) {
    cat("residual_new class:", class(result$residual_new), "\n")
  }

  # Time compiled
  system.time({
    for (i in 1:100) {
      result <- one_iteration_compiled(
        x_mlx, beta_mlx, residual_mlx, intercept_mlx,
        eta_mlx, y_mlx, ones_mlx, n, 0.01, 0.1, 1, is_gaussian
      )
      # Try just accessing it
      tmp <- result$residual_new
    }
  })

  cat("\nCompilation successful!\n")
}, error = function(e) {
  cat("Compilation or execution failed:\n")
  print(e)
})
