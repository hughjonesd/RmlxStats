
compare_error_sources <- function(n, eps) {
  x <- rnorm(n)
  y <- x + rnorm(n, 0, eps)
  fit_lm <- lm(y ~ x)
  fit_mlxs <- mlxs_lm(y ~ x)
  data.frame(
    n = n,
    eps = eps,
    comp_diff = abs(coef(fit_lm)["x"] - coef(fit_mlxs)["x"]),
    se = coef(summary(fit_lm))["x", 2] 
  )
}

params <- expand.grid(n = 10^(3:7), eps = 10^-(0:3))
results <- params |> 
  split(seq_len(nrow(params))) |> 
  lapply(\(r) compare_error_sources(r$n, r$eps))
results <- do.call(rbind, results)

library(tidyverse)
results |> 
  mutate(eps = factor(eps)) |> 
  ggplot(aes(n, comp_diff/se, group = eps, colour = eps)) +
    geom_line() +
    geom_hline(yintercept = c(0.1, 1), linetype = 2) +
    scale_x_log10() + 
    scale_y_log10() +
    theme_minimal()
  