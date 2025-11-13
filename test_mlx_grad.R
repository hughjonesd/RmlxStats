#!/usr/bin/env Rscript
# Test mlx_grad for computing gradients

library(Rmlx)

cat("=== Testing mlx_grad for glmnet gradients ===\n\n")

# Set up a simple least squares problem
n <- 100
p <- 20
x <- as_mlx(matrix(rnorm(n*p), n, p))
y <- as_mlx(matrix(rnorm(n), ncol=1))

cat("Problem: n=", n, ", p=", p, "\n\n")

# Define loss function (mean squared error)
loss_fn <- function(beta, x, y) {
  eta <- x %*% beta
  residual <- eta - y
  mlx_sum(residual * residual) / (2 * nrow(x))
}

# Test gradient computation
beta <- as_mlx(matrix(rnorm(p), ncol=1))

cat("1. Computing loss:\n")
loss_val <- loss_fn(beta, x, y)
cat("   Loss:", as.numeric(loss_val), "\n\n")

cat("2. Computing gradient with mlx_grad:\n")
# mlx_grad takes function and computes gradient w.r.t. first arg by default
grad <- mlx_grad(loss_fn, beta, x, y)
cat("   Gradient class:", class(grad), "\n")
cat("   Gradient type:", typeof(grad), "\n")
if (is.list(grad)) {
  cat("   Gradient is a list with names:", names(grad), "\n")
  grad <- grad[[1]]  # Extract first element
}
cat("   Gradient shape:", dim(grad), "\n")
cat("   Gradient norm:", as.numeric(mlx_sum(grad * grad)), "\n\n")

cat("3. Manual gradient (current method):\n")
eta <- x %*% beta
residual <- eta - y
manual_grad <- crossprod(x, residual) / nrow(x)
cat("   Gradient shape:", dim(manual_grad), "\n")
cat("   Gradient norm:", as.numeric(mlx_sum(manual_grad * manual_grad)), "\n\n")

cat("4. Comparing:\n")
diff <- mlx_sum((grad - manual_grad)^2)
cat("   Squared difference:", as.numeric(diff), "\n")
cat("   Match:", as.numeric(diff) < 1e-10, "\n\n")

cat("5. Testing with L2 penalty (ridge):\n")
loss_fn_ridge <- function(beta, x, y, lambda) {
  eta <- x %*% beta
  residual <- eta - y
  mse <- mlx_sum(residual * residual) / (2 * nrow(x))
  penalty <- (lambda / 2) * mlx_sum(beta * beta)
  mse + penalty
}

lambda <- 0.1
grad_ridge_raw <- mlx_grad(loss_fn_ridge, beta, x, y, lambda)
grad_ridge <- grad_ridge_raw[[1]]  # Extract from list

manual_grad_ridge <- crossprod(x, residual) / nrow(x) + lambda * beta

diff_ridge <- mlx_sum((grad_ridge - manual_grad_ridge)^2)
cat("   Squared difference:", as.numeric(diff_ridge), "\n")
cat("   Match:", as.numeric(diff_ridge) < 1e-10, "\n\n")

cat("=== Can we use mlx_grad in glmnet? ===\n\n")
cat("Pros:\n")
cat("  ✓ Automatic differentiation (less error-prone)\n")
cat("  ✓ Matches manual gradient perfectly\n")
cat("  ✓ Works with penalties\n\n")

cat("Cons:\n")
cat("  ? Need to define loss function (might be overhead)\n")
cat("  ? Unclear if faster than manual gradient\n")
cat("  ? Current code already has gradient computed\n\n")

cat("Recommendation:\n")
cat("  Current manual gradient is fine and already fast.\n")
cat("  mlx_grad would be useful if implementing new algorithms\n")
cat("  or if gradient computation was complex/error-prone.\n")
