#!/usr/bin/env Rscript
# Profile mlxs_glmnet to identify bottlenecks

library(RmlxStats)
library(Rmlx)
library(glmnet)
library(profvis)
library(microbenchmark)

set.seed(42)

# Test different problem sizes
configs <- list(
  small = list(n = 500, p = 50),
  medium = list(n = 2000, p = 100),
  large = list(n = 5000, p = 200)
)

cat("=== Timing Comparison ===\n\n")

for (name in names(configs)) {
  cfg <- configs[[name]]
  cat(sprintf("Problem size: %s (n=%d, p=%d)\n", name, cfg$n, cfg$p))

  x <- matrix(rnorm(cfg$n * cfg$p), nrow = cfg$n, ncol = cfg$p)
  beta_true <- c(runif(min(10, cfg$p), -1, 1), rep(0, cfg$p - min(10, cfg$p)))
  y <- drop(x %*% beta_true + rnorm(cfg$n))

  # Time glmnet
  t_glmnet <- system.time({
    fit_glmnet <- glmnet(x, y, family = "gaussian", alpha = 1,
                         nlambda = 20, standardize = TRUE)
  })

  # Time mlxs_glmnet - force evaluation
  t_mlxs <- system.time({
    fit_mlxs <- mlxs_glmnet(x, y, family = mlxs_gaussian(), alpha = 1,
                            nlambda = 20, standardize = TRUE)
    # beta is already converted to R, so just access it
    sum(fit_mlxs$beta)
  })

  cat(sprintf("  glmnet:     %.3fs\n", t_glmnet[3]))
  cat(sprintf("  mlxs_glmnet: %.3fs (%.1fx slower)\n",
              t_mlxs[3], t_mlxs[3]/t_glmnet[3]))
  cat("\n")
}

cat("\n=== Detailed Profiling ===\n\n")

# Profile a medium-sized problem
cfg <- configs$medium
x <- matrix(rnorm(cfg$n * cfg$p), nrow = cfg$n, ncol = cfg$p)
beta_true <- c(runif(min(10, cfg$p), -1, 1), rep(0, cfg$p - min(10, cfg$p)))
y <- drop(x %*% beta_true + rnorm(cfg$n))

cat("Running profvis on mlxs_glmnet (medium problem)...\n")
cat("Output will be saved to profile_mlxs_glmnet.html\n\n")

# Profile with fewer lambdas for clarity
p <- profvis::profvis({
  fit_mlxs <- mlxs_glmnet(x, y, family = mlxs_gaussian(), alpha = 1,
                          nlambda = 10, standardize = TRUE, maxit = 500)
  # beta is already converted
  sum(fit_mlxs$beta)
})

htmlwidgets::saveWidget(p, "profile_mlxs_glmnet.html")

cat("\n=== Breakdown by Operation ===\n\n")

# Manually time key operations
x_mlx <- Rmlx::as_mlx(x)
y_mlx <- Rmlx::as_mlx(matrix(y, ncol = 1))
beta_mlx <- Rmlx::as_mlx(matrix(0, nrow = cfg$p, ncol = 1))

cat("Single operation timings:\n")

# Matrix multiply
t1 <- system.time({
  for (i in 1:100) {
    result <- x_mlx %*% beta_mlx
    Rmlx::mlx_eval(result)
  }
})
cat(sprintf("  100x matrix multiply: %.3fs (%.3fms each)\n", t1[3], t1[3]*10))

# Crossprod
t2 <- system.time({
  for (i in 1:100) {
    result <- crossprod(x_mlx, y_mlx)
    Rmlx::mlx_eval(result)
  }
})
cat(sprintf("  100x crossprod: %.3fs (%.3fms each)\n", t2[3], t2[3]*10))

# Soft threshold
beta_test <- Rmlx::as_mlx(matrix(rnorm(cfg$p), ncol = 1))
t3 <- system.time({
  for (i in 1:100) {
    result <- RmlxStats:::.mlxs_soft_threshold(beta_test, 0.1)
    Rmlx::mlx_eval(result)
  }
})
cat(sprintf("  100x soft_threshold: %.3fs (%.3fms each)\n", t3[3], t3[3]*10))

# Link function (for GLM)
eta_test <- Rmlx::as_mlx(matrix(rnorm(cfg$n), ncol = 1))
fam <- mlxs_gaussian()
t4 <- system.time({
  for (i in 1:100) {
    result <- fam$linkinv(eta_test)
    Rmlx::mlx_eval(result)
  }
})
cat(sprintf("  100x linkinv: %.3fs (%.3fms each)\n", t4[3], t4[3]*10))

# as.numeric conversion
beta_mlx_test <- Rmlx::as_mlx(matrix(rnorm(cfg$p), ncol = 1))
t5 <- system.time({
  for (i in 1:100) {
    result <- as.numeric(beta_mlx_test)
  }
})
cat(sprintf("  100x as.numeric: %.3fs (%.3fms each)\n", t5[3], t5[3]*10))

cat("\n=== Memory Transfer Analysis ===\n\n")

# Count conversions in one lambda iteration
cat("Conversions per lambda iteration:\n")
cat("  R -> MLX: 1 (threshold value)\n")
cat("  MLX -> R: ~3-5 (delta_beta, intercept_grad, convergence checks)\n")
cat("  Per inner iteration (maxit): ~4-6 conversions\n")
cat("  For nlambda=100, maxit=1000: ~400,000 potential conversions\n")

cat("\n=== Complete ===\n")
