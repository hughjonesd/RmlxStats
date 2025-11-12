#!/usr/bin/env Rscript
# Benchmark Strategy 2 optimizations on large problems

library(RmlxStats)
library(Rmlx)
library(glmnet)

set.seed(42)

cat("=== Benchmark: Strategy 2 Optimizations ===\n\n")
cat("Testing on large problems as requested (n=5000, p=200)\n\n")

# Test configurations - focus on large problems
configs <- list(
  target = list(n = 5000, p = 200, nlambda = 100),
  target_small = list(n = 5000, p = 200, nlambda = 20),
  large1 = list(n = 10000, p = 100, nlambda = 50),
  large2 = list(n = 8000, p = 150, nlambda = 50)
)

results <- data.frame(
  config = character(),
  n = integer(),
  p = integer(),
  nlambda = integer(),
  glmnet_time = numeric(),
  mlxs_time = numeric(),
  speedup = numeric(),
  stringsAsFactors = FALSE
)

for (name in names(configs)) {
  cfg <- configs[[name]]
  cat(sprintf("Testing: %s (n=%d, p=%d, nlambda=%d)\n",
              name, cfg$n, cfg$p, cfg$nlambda))

  # Generate data
  x <- matrix(rnorm(cfg$n * cfg$p), nrow = cfg$n, ncol = cfg$p)
  beta_true <- c(runif(min(10, cfg$p), -1, 1), rep(0, cfg$p - min(10, cfg$p)))
  y <- drop(x %*% beta_true + rnorm(cfg$n))

  # Benchmark glmnet
  cat("  Running glmnet... ")
  t_glmnet <- system.time({
    fit_glmnet <- glmnet(x, y, family = "gaussian", alpha = 1,
                         nlambda = cfg$nlambda, standardize = TRUE)
  })
  cat(sprintf("%.3fs\n", t_glmnet[3]))

  # Benchmark mlxs_glmnet
  cat("  Running mlxs_glmnet... ")
  t_mlxs <- system.time({
    fit_mlxs <- mlxs_glmnet(x, y, family = mlxs_gaussian(), alpha = 1,
                            nlambda = cfg$nlambda, standardize = TRUE)
  })
  cat(sprintf("%.3fs\n", t_mlxs[3]))

  speedup <- t_glmnet[3] / t_mlxs[3]
  cat(sprintf("  Speedup vs baseline: %.2fx %s\n\n",
              speedup,
              if(speedup > 1) "FASTER" else "slower"))

  results <- rbind(results, data.frame(
    config = name,
    n = cfg$n,
    p = cfg$p,
    nlambda = cfg$nlambda,
    glmnet_time = t_glmnet[3],
    mlxs_time = t_mlxs[3],
    speedup = speedup,
    stringsAsFactors = FALSE
  ))
}

cat("\n=== Summary Table ===\n\n")
print(results, row.names = FALSE)

cat("\n=== Analysis ===\n\n")
cat(sprintf("Target problem (n=5000, p=200): %.2fx vs glmnet\n",
            results$speedup[results$config == "target"]))

avg_speedup <- mean(results$speedup)
cat(sprintf("Average speedup across all tests: %.2fx\n", avg_speedup))

if (avg_speedup > 1) {
  cat("\n*** SUCCESS: mlxs_glmnet is now FASTER than glmnet! ***\n")
} else {
  cat(sprintf("\nmlxs_glmnet is still %.2fx slower than glmnet on average\n",
              1/avg_speedup))
  cat("But this is a significant improvement from 10-155x slower baseline.\n")
}

cat("\n=== Complete ===\n")
