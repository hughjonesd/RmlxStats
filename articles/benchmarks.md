# Benchmarks

We benchmark RmlxStats against standard R functions and specialized fast
fitting packages across varying dataset sizes of cases (`n`) and
predictors (`p`).

## Data Generation

``` r
set.seed(20251111)

n_max <- 50000
p_max <- 200

X <- matrix(rnorm(n_max * p_max), nrow = n_max, ncol = p_max)
colnames(X) <- paste0("x", seq_len(p_max))

beta_true <- rnorm(p_max, mean = 0, sd = 0.5)
y_continuous <- drop(X %*% beta_true + rnorm(n_max, sd = 2))
linpred <- drop(X %*% beta_true) / 5  
prob <- 1 / (1 + exp(-linpred))
y_binary <- rbinom(n_max, size = 1, prob = prob)

full_data <- data.frame(
  y_cont = y_continuous,
  y_bin = y_binary,
  X
)

dim(full_data)
#> [1] 50000   202

n_sizes <- c(2000, 10000, 50000)
p_sizes <- c(50, 100, 200)

bench_grid <- expand.grid(
  n = n_sizes,
  p = p_sizes,
  stringsAsFactors = FALSE
)

bench_grid <- bench_grid[bench_grid$n > bench_grid$p, ]

bench_grid
#>       n   p
#> 1  2000  50
#> 2 10000  50
#> 3 50000  50
#> 4  2000 100
#> 5 10000 100
#> 6 50000 100
#> 7  2000 200
#> 8 10000 200
#> 9 50000 200
```

## Linear Model Benchmarks

``` r
lm_results <- list()

for (i in seq_len(nrow(bench_grid))) {
  n <- bench_grid$n[i]
  p <- bench_grid$p[i]

  subset_data <- full_data[1:n, c("y_cont", paste0("x", 1:p))]
  formula_str <- paste("y_cont ~", paste(paste0("x", 1:p), collapse = " + "))
  lm_formula <- as.formula(formula_str)

  bm <- mark(
    lm = lm(lm_formula, data = subset_data),
    mlxs_lm = {
      l <- mlxs_lm(lm_formula, data = subset_data)
      Rmlx::mlx_eval(l$coefficients)
    },
    feols = feols(lm_formula, data = subset_data),
    fastLm = RcppEigen::fastLm(lm_formula, data = subset_data),
    speedlm = speedglm::speedlm(lm_formula, data = subset_data),
    iterations = 3,
    check = FALSE,
    filter_gc = FALSE
  )

  bm$n <- n
  bm$p <- p
  bm$model_type <- "LM"
  lm_results[[i]] <- bm
}

lm_df <- bind_rows(lm_results)
```

## GLM Benchmarks

``` r
glm_results <- list()

for (i in seq_len(nrow(bench_grid))) {
  n <- bench_grid$n[i]
  p <- bench_grid$p[i]

  subset_data <- full_data[1:n, c("y_bin", paste0("x", 1:p))]
  formula_str <- paste("y_bin ~", paste(paste0("x", 1:p), collapse = " + "))
  glm_formula <- as.formula(formula_str)

  bm <- mark(
    glm = glm(glm_formula, family = binomial(),
                             data = subset_data,
                             control = list(maxit = 50)),
    mlxs_glm = {
      g <- mlxs_glm(glm_formula, family = mlxs_binomial(),
                        data = subset_data,
                        control = list(maxit = 50, epsilon = 1e-6))
      Rmlx::mlx_eval(g$coefficients)
    },
    speedglm = speedglm::speedglm(glm_formula, family = binomial(),
                                   data = subset_data),
    iterations = 3,
    check = FALSE,
    filter_gc = FALSE
  )

  bm$n <- n
  bm$p <- p
  bm$model_type <- "GLM"
  glm_results[[i]] <- bm
}

glm_df <- bind_rows(glm_results)
```

## Bootstrap Benchmarks

For bootstrap, we use smaller datasets and fewer bootstrap iterations
due to computational cost.

``` r
boot_grid <- expand.grid(
  n = c(2000, 10000),
  p = c(50, 100),
  stringsAsFactors = FALSE
)

boot_results <- list()

for (i in seq_len(nrow(boot_grid))) {
  n <- boot_grid$n[i]
  p <- boot_grid$p[i]

  subset_data <- full_data[1:n, c("y_cont", paste0("x", 1:p))]
  formula_str <- paste("y_cont ~", paste(paste0("x", 1:p), collapse = " + "))
  boot_formula <- as.formula(formula_str)

  fit_mlxs <- mlxs_lm(boot_formula, data = subset_data)
  fit_base <- lm(boot_formula, data = subset_data)

  # Bootstrap function for boot package
  boot_stat <- function(dat, idx) {
    coef(lm(boot_formula, data = dat[idx, , drop = FALSE]))
  }

  bm <- mark(
    boot_case = boot::boot(subset_data, statistic = boot_stat,
                          R = 50L, parallel = "no"),
    lmboot_case = lmboot::paired.boot(boot_formula, data = subset_data, 
                                      B = 50L),
    lmboot_resid = lmboot::residual.boot(boot_formula, data = subset_data, 
                                         B = 50L),
    mlxs_case = {
      s <- summary(fit_mlxs, bootstrap = TRUE,
              bootstrap_args = list(B = 50L, seed = 42,
                                   bootstrap_type = "case",
                                   progress = FALSE))
      Rmlx::mlx_eval(s$std.err)
    },
    mlxs_resid = {
      s <- summary(fit_mlxs, bootstrap = TRUE,
              bootstrap_args = list(B = 50L, seed = 42,
                                   bootstrap_type = "resid",
                                   progress = FALSE))
      Rmlx::mlx_eval(s$std.err)
    },
    iterations = 3,
    check = FALSE,
    filter_gc = FALSE,
    memory = FALSE
  )

  bm$n <- n
  bm$p <- p
  bm$model_type <- "Bootstrap"
  boot_results[[i]] <- bm
}
#> Warning in lmboot::paired.boot(boot_formula, data = subset_data, B = 50L): Number of bootstrap samples is recommended to be more than the number of observations.
#> Warning in lmboot::paired.boot(boot_formula, data = subset_data, B = 50L): Number of bootstrap samples is recommended to be more than the number of observations.
#> Warning in lmboot::paired.boot(boot_formula, data = subset_data, B = 50L): Number of bootstrap samples is recommended to be more than the number of observations.
#> Warning in lmboot::residual.boot(boot_formula, data = subset_data, B = 50L): Number of bootstrap samples is recommended to be more than the number of observations.
#> Warning in lmboot::residual.boot(boot_formula, data = subset_data, B = 50L): Number of bootstrap samples is recommended to be more than the number of observations.
#> Warning in lmboot::residual.boot(boot_formula, data = subset_data, B = 50L): Number of bootstrap samples is recommended to be more than the number of observations.
#> Warning in lmboot::paired.boot(boot_formula, data = subset_data, B = 50L): Number of bootstrap samples is recommended to be more than the number of observations.
#> Warning in lmboot::paired.boot(boot_formula, data = subset_data, B = 50L): Number of bootstrap samples is recommended to be more than the number of observations.
#> Warning in lmboot::paired.boot(boot_formula, data = subset_data, B = 50L): Number of bootstrap samples is recommended to be more than the number of observations.
#> Warning in lmboot::residual.boot(boot_formula, data = subset_data, B = 50L): Number of bootstrap samples is recommended to be more than the number of observations.
#> Warning in lmboot::residual.boot(boot_formula, data = subset_data, B = 50L): Number of bootstrap samples is recommended to be more than the number of observations.
#> Warning in lmboot::residual.boot(boot_formula, data = subset_data, B = 50L): Number of bootstrap samples is recommended to be more than the number of observations.
#> Warning in lmboot::paired.boot(boot_formula, data = subset_data, B = 50L): Number of bootstrap samples is recommended to be more than the number of observations.
#> Warning in lmboot::paired.boot(boot_formula, data = subset_data, B = 50L): Number of bootstrap samples is recommended to be more than the number of observations.
#> Warning in lmboot::paired.boot(boot_formula, data = subset_data, B = 50L): Number of bootstrap samples is recommended to be more than the number of observations.
#> Warning in lmboot::residual.boot(boot_formula, data = subset_data, B = 50L): Number of bootstrap samples is recommended to be more than the number of observations.
#> Warning in lmboot::residual.boot(boot_formula, data = subset_data, B = 50L): Number of bootstrap samples is recommended to be more than the number of observations.
#> Warning in lmboot::residual.boot(boot_formula, data = subset_data, B = 50L): Number of bootstrap samples is recommended to be more than the number of observations.
#> Warning in lmboot::paired.boot(boot_formula, data = subset_data, B = 50L): Number of bootstrap samples is recommended to be more than the number of observations.
#> Warning in lmboot::paired.boot(boot_formula, data = subset_data, B = 50L): Number of bootstrap samples is recommended to be more than the number of observations.
#> Warning in lmboot::paired.boot(boot_formula, data = subset_data, B = 50L): Number of bootstrap samples is recommended to be more than the number of observations.
#> Warning in lmboot::residual.boot(boot_formula, data = subset_data, B = 50L): Number of bootstrap samples is recommended to be more than the number of observations.
#> Warning in lmboot::residual.boot(boot_formula, data = subset_data, B = 50L): Number of bootstrap samples is recommended to be more than the number of observations.
#> Warning in lmboot::residual.boot(boot_formula, data = subset_data, B = 50L): Number of bootstrap samples is recommended to be more than the number of observations.

boot_df <- bind_rows(boot_results)
```

## Visualization

``` r
all_results <- bind_rows(lm_df, glm_df, boot_df)

all_results$method <- as.character(all_results$expression)
all_results$median_sec <- as.numeric(all_results$median, units = "s")
all_results$is_mlxs <- grepl("^mlxs", all_results$method)

# Summary statistics
summary_results <- all_results |>
  filter(!is.na(median_sec)) |>
  group_by(model_type, n, p, method, is_mlxs) |>
  summarise(
    median_sec = median(median_sec),
    .groups = "drop"
  )
```

### Linear Models

``` r
lm_data <- summary_results |>
  filter(model_type == "LM")

ggplot(lm_data, aes(x = method, y = median_sec, fill = is_mlxs)) +
  geom_col() +
  facet_grid(n ~ p, scales = "free", labeller = "label_both") +
  scale_fill_manual(
    values = c("TRUE" = "#E74C3C", "FALSE" = "#95A5A6"),
    labels = c("TRUE" = "RmlxStats", "FALSE" = "Other"),
    name = "Package"
  ) +
  labs(
    title = "Linear Model Benchmark: Time by Dataset Size",
    x = "Method",
    y = "Median Time (seconds)",
    caption = "Lower is better"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(face = "bold"),
    legend.position = "top"
  ) +
  coord_flip()
```

![](benchmarks_files/figure-html/plot-lm-1.png)

### GLM Models

``` r
glm_data <- summary_results |>
  filter(model_type == "GLM")

ggplot(glm_data, aes(x = method, y = median_sec, fill = is_mlxs)) +
  geom_col() +
  facet_grid(n ~ p, scales = "free", labeller = "label_both") +
  scale_fill_manual(
    values = c("TRUE" = "#E74C3C", "FALSE" = "#95A5A6"),
    labels = c("TRUE" = "RmlxStats", "FALSE" = "Other"),
    name = "Package"
  ) +
  labs(
    title = "GLM Benchmark: Time by Dataset Size",
    x = "Method",
    y = "Median Time (seconds)",
    caption = "Lower is better"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(face = "bold"),
    legend.position = "top"
  ) +
  coord_flip()
```

![](benchmarks_files/figure-html/plot-glm-1.png)

### Bootstrap

``` r
boot_data <- summary_results |>
  filter(model_type == "Bootstrap")

boot_data$method <- factor(boot_data$method, 
                           levels = c("mlxs_case", "lmboot_case", "boot_case",
                                      "mlxs_resid", "lmboot_resit"))
ggplot(boot_data, aes(x = method, y = median_sec, fill = is_mlxs)) +
  geom_col() +
  facet_grid(n ~ p, scales = "free", labeller = "label_both") +
  scale_fill_manual(
    values = c("TRUE" = "#E74C3C", "FALSE" = "#95A5A6"),
    labels = c("TRUE" = "RmlxStats", "FALSE" = "Other"),
    name = "Package"
  ) +
  labs(
    title = "Bootstrap Benchmark: Time by Dataset Size",
    x = "Method",
    y = "Median Time (seconds)",
    caption = "Lower is better. Bootstrap with 50 iterations."
  ) +
  theme_minimal(base_size = 11) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(face = "bold"),
    legend.position = "top"
  ) +
  coord_flip()
```

![](benchmarks_files/figure-html/plot-bootstrap-1.png)
