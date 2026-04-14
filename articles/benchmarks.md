# Benchmarks

We benchmark RmlxStats against base R and specialized fast fitting
packages, across varying numbers of cases (`n`) and predictors (`p`).

Benchmarking was run on an M2 Macbook Air.

## Data Generation

``` r
set.seed(20251111)

n_max <- 50000
p_max <- 200

X <- matrix(rnorm(n_max * p_max), nrow = n_max, ncol = p_max)
colnames(X) <- paste0("x", seq_len(p_max))

beta_true <- rnorm(p_max, mean = 0, sd = 0.5)
y_continuous <- drop(X %*% beta_true + rnorm(n_max, sd = 2))

# only 1 in 10 predictors matter:
X_sparse <- X[, seq(10, p_max, 10)]
beta_true_sparse <- beta_true[seq(1, p_max, 10)]
y_sparse <- drop(X_sparse %*% beta_true_sparse + rnorm(n_max, sd = 2))

linpred <- drop(X %*% beta_true) / 5  
prob <- 1 / (1 + exp(-linpred))
y_binary <- rbinom(n_max, size = 1, prob = prob)

full_data <- data.frame(
  y_cont = y_continuous,
  y_bin = y_binary,
  y_sparse = y_sparse,
  X
)

n_sizes <- c(2000, 10000, 50000)
p_sizes <- c(25, 50, 100, 200)

# for fast debugging
if (params$develop) {
  p_sizes <- p_sizes/10
  n_sizes <- n_sizes/10
}

bench_grid <- expand.grid(
  n = n_sizes,
  p = p_sizes,
  stringsAsFactors = FALSE
)

bench_grid <- bench_grid[bench_grid$n > bench_grid$p, ]
```

## `lm` Benchmarks

``` r
lm_results <- list()

for (i in seq_len(nrow(bench_grid))) {
  n <- bench_grid$n[i]
  p <- bench_grid$p[i]

  subset_data <- full_data[1:n, c("y_cont", paste0("x", 1:p))]
  formula_str <- paste("y_cont ~", paste(paste0("x", 1:p), collapse = " + "))
  lm_formula <- as.formula(formula_str)

  bm <- mark(
    "stats::lm" = lm(lm_formula, data = subset_data),
    "RmlxStats::mlxs_lm" = {
      l <- mlxs_lm(lm_formula, data = subset_data)
      Rmlx::mlx_eval(l$coefficients)
    },
    "fixest::feols" = feols(lm_formula, data = subset_data),
    "RcppEigen::fastLm" = RcppEigen::fastLm(lm_formula, data = subset_data),
    "speedglm::speedlm" = speedglm::speedlm(lm_formula, data = subset_data),
    iterations = 3,
    check = FALSE,
    filter_gc = FALSE
  )

  bm$n <- n
  bm$p <- p
  bm$model_type <- "lm"
  lm_results[[i]] <- bm
}

lm_df <- do.call(rbind, lm_results)
```

## `glm` Benchmarks

``` r
glm_results <- list()

for (i in seq_len(nrow(bench_grid))) {
  n <- bench_grid$n[i]
  p <- bench_grid$p[i]

  subset_data <- full_data[1:n, c("y_bin", paste0("x", 1:p))]
  formula_str <- paste("y_bin ~", paste(paste0("x", 1:p), collapse = " + "))
  glm_formula <- as.formula(formula_str)

  bm <- mark(
    "stats::glm" = glm(glm_formula, family = binomial(),
                             data = subset_data,
                             control = list(maxit = 50)),
    "RmlxStats::mlxs_glm" = {
      g <- mlxs_glm(glm_formula, family = mlxs_binomial(),
                        data = subset_data,
                        control = list(maxit = 50, epsilon = 1e-6))
      Rmlx::mlx_eval(g$coefficients)
    },
    "speedglm::speedglm" = speedglm::speedglm(glm_formula, family = binomial(),
                                   data = subset_data),
    iterations = 3,
    check = FALSE,
    filter_gc = FALSE
  )

  bm$n <- n
  bm$p <- p
  bm$model_type <- "glm"
  glm_results[[i]] <- bm
}

glm_df <- do.call(rbind, glm_results)
```

## `glmnet` Benchmarks

``` r
glmnet_results <- list()

for (i in seq_len(nrow(bench_grid))) {
  n <- bench_grid$n[i]
  p <- bench_grid$p[i]

  xvars <- paste0("x", 1:p)
  x <- full_data[1:n, xvars]
  y <- full_data[1:n, "y_sparse"]

  bm <- mark(
    "glmnet::glmnet" = glmnet::glmnet(x, y, lambda = 1/1:50),
    "RmlxStats::mlxs_glmnet" = mlxs_glmnet(x, y, lambda = 1/1:50),
    iterations = 3,
    check = FALSE,
    filter_gc = FALSE
  )

  bm$n <- n
  bm$p <- p
  bm$model_type <- "glmnet"
  glmnet_results[[i]] <- bm
}

glmnet_df <- do.call(rbind, glmnet_results)
```

## Bootstrap Benchmarks

For bootstrap, we use only the smaller datasets due to computational
cost.

``` r
boot_grid <- expand.grid(
  n = n_sizes[1:2],
  p = p_sizes[1:3],
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

boot_df <- do.call(rbind, boot_results)
```

## Summary Tables

We compare RmlxStats functions both against base R, and against the
fastest alternative tested. Numbers show
`RmlxStats time/alternative time`.

|  | n=2000, p=25 | n=10000, p=25 | n=50000, p=25 | n=2000, p=50 | n=10000, p=50 | n=50000, p=50 |
|---:|---:|---:|---:|---:|---:|---:|
| Linear Model | 87.3 | 72.0 | 57.3 | 65.5 | 44.3 | 36.9 |
| Logistic Regression | 201.9 | 67.7 | 34.1 | 95.2 | 36.7 | 22.7 |
| Elastic Net | 7637.9 | 2549.7 | 542.3 | 4132.5 | 1344.6 | 287.3 |
| Bootstrap (Case) | 105.2 | 42.5 |    | 51.7 | 27.7 |    |
| Bootstrap (Residual) | 710.8 | 202.0 |    | 325.0 | 106.7 |    |
|  | n=2000, p=100 | n=10000, p=100 | n=50000, p=100 | n=2000, p=200 | n=10000, p=200 | n=50000, p=200 |
| Linear Model | 41.4 | 25.6 | 22.9 | 26.6 | 15.4 | 12.1 |
| Logistic Regression | 43.5 | 20.4 | 12.9 | 18.4 | 12.1 | 10.1 |
| Elastic Net | 2055.4 | 689.7 | 143.6 | 857.0 | 317.4 | 106.5 |
| Bootstrap (Case) | 28.4 | 17.4 |    |    |    |  |
| Bootstrap (Residual) | 123.8 | 36.7 |    |    |    |  |

RmlxStats time vs base R (%). Below 100% → RmlxStats is faster
{#tab:summary-tables}

|  | n=2000, p=25 | n=10000, p=25 | n=50000, p=25 | n=2000, p=50 | n=10000, p=50 | n=50000, p=50 |
|---:|---:|---:|---:|---:|---:|---:|
| Linear Model | 110.0 | 97.1 | 72.2 | 109.2 | 72.4 | 60.3 |
| Logistic Regression | 256.1 | 96.7 | 47.8 | 119.8 | 47.1 | 29.5 |
| Elastic Net | 7637.9 | 2549.7 | 542.3 | 4132.5 | 1344.6 | 287.3 |
| Bootstrap (Case) | 133.1 | 42.5 |    | 51.7 | 27.7 |    |
| Bootstrap (Residual) | 710.8 | 202.0 |    | 325.0 | 106.7 |    |
|  | n=2000, p=100 | n=10000, p=100 | n=50000, p=100 | n=2000, p=200 | n=10000, p=200 | n=50000, p=200 |
| Linear Model | 81.1 | 55.9 | 45.4 | 64.2 | 34.5 | 26.7 |
| Logistic Regression | 50.3 | 25.3 | 16.6 | 19.8 | 14.5 | 12.3 |
| Elastic Net | 2055.4 | 689.7 | 143.6 | 857.0 | 317.4 | 106.5 |
| Bootstrap (Case) | 28.4 | 17.4 |    |    |    |  |
| Bootstrap (Residual) | 123.8 | 36.7 |    |    |    |  |

RmlxStats time vs fastest alternative (%). Below 100% → RmlxStats is
faster {#tab:summary-tables}

## `lm` Models

![](benchmarks_files/figure-html/plot-lm-1.png)

## `glm` Models

![](benchmarks_files/figure-html/plot-glm-1.png)

## `glmnet` Models

![](benchmarks_files/figure-html/plot-glmnet-1.png)

## Bootstraps

![](benchmarks_files/figure-html/plot-bootstrap-1.png)
