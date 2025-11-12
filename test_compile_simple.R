#!/usr/bin/env Rscript
# Simpler test: return just one value

library(Rmlx)

# One iteration, returning just the residual
one_iteration_simple <- function(x_active, beta_prev, residual, y, n_obs, step, lambda_val, alpha) {
  grad <- crossprod(x_active, residual) / n_obs
  grad <- grad + beta_prev * (lambda_val * (1 - alpha))

  beta_temp <- beta_prev - grad * step
  thresh <- lambda_val * alpha * step

  # Soft threshold
  abs_beta <- abs(beta_temp)
  magnitude <- mlx_maximum(abs_beta - thresh, 0)
  sign_beta <- beta_temp / (abs_beta + 1e-10)
  beta_new <- magnitude * sign_beta

  delta <- beta_new - beta_prev
  eta_new <- x_active %*% beta_new
  residual_new <- eta_new - y

  residual_new
}

set.seed(42)
n <- 1000
p <- 100
x <- as_mlx(matrix(rnorm(n*p), n, p))
y <- as_mlx(matrix(rnorm(n), ncol=1))
beta <- mlx_zeros(c(p, 1))
residual <- x %*% beta - y

cat("Uncompiled version:\n")
t1 <- system.time({
  for (i in 1:1000) {
    result <- one_iteration_simple(x, beta, residual, y, n, 0.01, 0.1, 1)
    mlx_eval(result)
  }
})
print(t1)

cat("\nCompiling...\n")
one_iteration_compiled <- mlx_compile(one_iteration_simple)

cat("Compiled version:\n")
t2 <- system.time({
  for (i in 1:1000) {
    result <- one_iteration_compiled(x, beta, residual, y, n, 0.01, 0.1, 1)
    mlx_eval(result)
  }
})
print(t2)

cat(sprintf("\nSpeedup: %.2fx\n", t1[3]/t2[3]))
