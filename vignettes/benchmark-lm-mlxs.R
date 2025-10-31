## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

has_packages <- requireNamespace("Rmlx", quietly = TRUE) &&
  requireNamespace("nycflights13", quietly = TRUE) &&
  requireNamespace("bench", quietly = TRUE) &&
  requireNamespace("fixest", quietly = TRUE) &&
  requireNamespace("RcppEigen", quietly = TRUE) &&
  requireNamespace("speedglm", quietly = TRUE)

if (!has_packages) {
  knitr::opts_chunk$set(eval = FALSE)
  message("Skipping code evaluation: missing Rmlx, nycflights13, bench, fixest, or RcppEigen.")
} else {
  library(RmlxStats)
  library(Rmlx)
  library(nycflights13)
  library(bench)
  library(fixest)
  library(RcppEigen)
  library(speedglm)
}

## ----data-prep----------------------------------------------------------------
flights <- as.data.frame(nycflights13::flights)
vars <- c("arr_delay", "dep_delay", "air_time", "distance")
complete_rows <- complete.cases(flights[, vars])
bench_data <- flights[complete_rows, vars]
nrow(bench_data)

## ----timings, cache=FALSE-----------------------------------------------------
lm_formula <- arr_delay ~ dep_delay + air_time + distance

bench_mark <- mark(
  lm = lm(lm_formula, data = bench_data),
  mlxs_lm = mlxs_lm(lm_formula, data = bench_data),
  feols = feols(lm_formula, data = bench_data),
  fastLm = RcppEigen::fastLm(lm_formula, data = bench_data),
  speedlm = speedglm::speedlm(lm_formula, data = bench_data),
  iterations = 5,
  check = FALSE
)

# Summarize median timings, memory allocation, and relative speed.
bench_summary <- data.frame(
  method = as.character(bench_mark$expression),
  median_sec = as.numeric(bench_mark$median, units = "s"),
  mem_mb = as.numeric(bench_mark$mem_alloc, units = "MB"),
  itr_per_sec = bench_mark$`itr/sec`
)
bench_summary$relative <- bench_summary$median_sec / min(bench_summary$median_sec)
bench_summary

## ----agreement----------------------------------------------------------------
lm_fit <- lm(lm_formula, data = bench_data)
mlxs_fit <- mlxs_lm(lm_formula, data = bench_data)
feols_fit <- feols(lm_formula, data = bench_data)
fastlm_fit <- RcppEigen::fastLm(lm_formula, data = bench_data)
speedlm_fit <- speedglm::speedlm(lm_formula, data = bench_data)

coef_delta <- max(abs(coef(lm_fit) - mlxs_fit$coefficients))
fitted_delta <- max(abs(fitted(lm_fit) - mlxs_fit$fitted.values))

feols_coef_delta <- max(abs(coef(lm_fit) - coef(feols_fit)))
feols_fitted_delta <- max(abs(fitted(lm_fit) - as.vector(predict(feols_fit))))

fastlm_coef_delta <- max(abs(coef(lm_fit) - fastlm_fit$coefficients))
fastlm_fitted_delta <- max(abs(fitted(lm_fit) - fastlm_fit$fitted.values))

speedlm_coef_delta <- max(abs(coef(lm_fit) - speedlm_fit$coefficients))
speedlm_fitted_delta <- max(abs(fitted(lm_fit) - as.vector(predict(speedlm_fit))))

c(
  max_coefficient_difference = coef_delta,
  max_fitted_difference = fitted_delta,
  feols_max_coefficient_difference = feols_coef_delta,
  feols_max_fitted_difference = feols_fitted_delta,
  fastlm_max_coefficient_difference = fastlm_coef_delta,
  fastlm_max_fitted_difference = fastlm_fitted_delta,
  speedlm_max_coefficient_difference = speedlm_coef_delta,
  speedlm_max_fitted_difference = speedlm_fitted_delta
)

## ----highp-data---------------------------------------------------------------
set.seed(20251031)
n_hd <- 8000
p_hd <- 150
x_hd <- matrix(rnorm(n_hd * p_hd), nrow = n_hd, ncol = p_hd)
colnames(x_hd) <- paste0("x", seq_len(p_hd))
beta_true <- runif(p_hd, -1, 1)
y_hd <- drop(x_hd %*% beta_true + rnorm(n_hd, sd = 0.5))
hd_data <- data.frame(y = y_hd, x_hd)
hd_formula <- as.formula(paste("y ~", paste(colnames(x_hd), collapse = " + ")))

## ----highp-benchmark, cache=FALSE---------------------------------------------
hd_mark <- mark(
  lm = lm(hd_formula, data = hd_data),
  mlxs_lm = mlxs_lm(hd_formula, data = hd_data),
  feols = feols(hd_formula, data = hd_data),
  fastLm = RcppEigen::fastLm(hd_formula, data = hd_data),
  speedlm = speedglm::speedlm(hd_formula, data = hd_data),
  iterations = 1,
  check = FALSE
)

hd_summary <- data.frame(
  method = as.character(hd_mark$expression),
  median_sec = as.numeric(hd_mark$median, units = "s"),
  mem_mb = as.numeric(hd_mark$mem_alloc, units = "MB"),
  itr_per_sec = hd_mark$`itr/sec`
)
hd_summary$relative <- hd_summary$median_sec / min(hd_summary$median_sec)
hd_summary

## ----highp-agreement----------------------------------------------------------
lm_hd <- lm(hd_formula, data = hd_data)
mlxs_hd <- mlxs_lm(hd_formula, data = hd_data)
speedlm_hd <- speedglm::speedlm(hd_formula, data = hd_data)

c(
  max(abs(coef(lm_hd) - mlxs_hd$coefficients)),
  max(abs(coef(lm_hd) - speedlm_hd$coefficients))
)

