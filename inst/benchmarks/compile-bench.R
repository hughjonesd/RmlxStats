#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(RmlxStats)
  library(Rmlx)
  library(bench)
})

set.seed(20251110)
n <- 20000L
p <- 256L
x_mat <- matrix(rnorm(n * p), nrow = n)
beta <- runif(p + 1L, -0.5, 0.5)
y_gauss <- beta[1L] + x_mat %*% beta[-1L] + rnorm(n)
prob <- 1 / (1 + exp(-y_gauss))
y_binom <- rbinom(n, size = 1, prob = prob)
colnames(x_mat) <- sprintf("x%03d", seq_len(p))
gauss_df <- data.frame(y = as.numeric(y_gauss), x_mat)
binom_df <- data.frame(y = as.integer(y_binom), x_mat)
formula_all <- y ~ .

bench_once <- function(enable_compile) {
  if (enable_compile) {
    Rmlx::mlx_enable_compile()
  } else {
    Rmlx::mlx_disable_compile()
  }
  label <- if (enable_compile) "compile_on" else "compile_off"

  lm_bench <- bench::mark(
    mlxs = mlxs_lm(formula_all, data = gauss_df),
    iterations = 3,
    check = FALSE
  )

  glm_bench <- bench::mark(
    mlxs = mlxs_glm(formula_all, family = mlxs_binomial(), data = binom_df,
                    control = list(maxit = 50, epsilon = 1e-6)),
    iterations = 3,
    check = FALSE
  )

  lm_fit <- mlxs_lm(formula_all, data = gauss_df)
  boot_bench <- bench::mark(
    summary(
      lm_fit,
      bootstrap = TRUE,
      bootstrap_args = list(B = 40L, seed = 2025, bootstrap_type = "case", progress = FALSE)
    ),
    iterations = 3,
    check = FALSE
  )

  data.frame(
    label = label,
    lm_median = as.numeric(lm_bench$median, units = "s"),
    glm_median = as.numeric(glm_bench$median, units = "s"),
    boot_median = as.numeric(boot_bench$median, units = "s"),
    row.names = NULL
  )
}

results <- do.call(rbind, lapply(c(FALSE, TRUE), bench_once))
print(results)
