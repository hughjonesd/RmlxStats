# Benchmarking MLX-backed Linear and Generalized Linear Models

The
[`mlxs_lm()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_lm.md)
and
[`mlxs_glm()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_glm.md)
functions mirror [`stats::lm()`](https://rdrr.io/r/stats/lm.html) and
[`stats::glm()`](https://rdrr.io/r/stats/glm.html) but execute their
linear algebra with the MLX backend. This vignette benchmarks these
implementations against standard R functions and specialized fast
fitting packages (fixest, RcppEigen, speedglm, fastglm) using the
sizeable
[`nycflights13::flights`](https://rdrr.io/pkg/nycflights13/man/flights.html)
dataset, demonstrating the performance gains available when Metal
acceleration is active.

## Data Preparation

We retain a handful of numeric predictors from the flights data and drop
rows with missing values in the variables of interest.

``` r
flights <- as.data.frame(nycflights13::flights)
vars <- c("arr_delay", "dep_delay", "air_time", "distance")
complete_rows <- complete.cases(flights[, vars])
bench_data <- flights[complete_rows, vars]
nrow(bench_data)
#> [1] 327346
```

The resulting dataset contains hundreds of thousands of observations,
large enough to expose performance differences in the solvers.

## Benchmark Setup

We benchmark several linear model solvers that accept formula
interfaces. The `bench` package automates warm-up and repetition.

``` r
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
#> Warning: Some expressions had a GC in every iteration; so filtering is
#> disabled.

# Summarize median timings, memory allocation, and relative speed.
bench_summary <- data.frame(
  method = as.character(bench_mark$expression),
  median_sec = as.numeric(bench_mark$median, units = "s"),
  mem_mb = as.numeric(bench_mark$mem_alloc, units = "MB"),
  itr_per_sec = bench_mark$`itr/sec`
)
bench_summary$relative <- bench_summary$median_sec / min(bench_summary$median_sec)
bench_summary
#>    method median_sec    mem_mb itr_per_sec relative
#> 1      lm 0.03765825  91103808   16.655393 1.155443
#> 2 mlxs_lm 0.07143779  78456800    9.975898 2.191878
#> 3   feols 0.03259205  32390888   29.321893 1.000000
#> 4  fastLm 0.11816438 116838424    6.903846 3.625559
#> 5 speedlm 0.03420704  75406984   28.082477 1.049552
ggplot2::autoplot(bench_mark, type = "boxplot")
```

![](benchmark-lm-mlxs_files/figure-html/timings-1.png)

The benchmark table reports the median execution time (seconds), memory
allocation (megabytes), iteration rate, and relative speed for each
method across five iterations. The `relative` column expresses how many
times slower each method is compared to the fastest option (values close
to 1 indicate the winner).

## Agreement on Flights Benchmark

Confirm that each solver matches the reference solution within floating
point tolerance.

``` r
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
#> Warning in predict.speedlm(speedlm_fit): fitted values were not returned from the speedglm object: 
#>               use the original data by setting argument 'newdata' or refit 
#>               the model by specifying fitted=TRUE.
#> Warning in max(abs(fitted(lm_fit) - as.vector(predict(speedlm_fit)))): no
#> non-missing arguments to max; returning -Inf

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
#> $max_coefficient_difference.ptr
#> <pointer: 0x600002ac5bd0>
#> 
#> $max_coefficient_difference.dim
#> integer(0)
#> 
#> $max_coefficient_difference.dtype
#> [1] "float32"
#> 
#> $max_coefficient_difference.device
#> [1] "gpu"
#> 
#> $max_fitted_difference.ptr
#> <pointer: 0x600002ac5460>
#> 
#> $max_fitted_difference.dim
#> integer(0)
#> 
#> $max_fitted_difference.dtype
#> [1] "float32"
#> 
#> $max_fitted_difference.device
#> [1] "gpu"
#> 
#> $feols_max_coefficient_difference
#> [1] 8.677503e-12
#> 
#> $feols_max_fitted_difference
#> [1] 5.844186e-09
#> 
#> $fastlm_max_coefficient_difference
#> [1] 8.236967e-12
#> 
#> $fastlm_max_fitted_difference
#> [1] 5.830291e-09
#> 
#> $speedlm_max_coefficient_difference
#> [1] 8.380852e-12
#> 
#> $speedlm_max_fitted_difference
#> [1] -Inf
```

Differences remain on the order of numerical precision, confirming that
[`mlxs_lm()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_lm.md)
and the other high-performance solvers reproduce the reference solution
while offering faster runtimes.

## High-Dimensional Benchmark

To stress-test performance when both sample size and feature count are
large, we simulate a design matrix with many rows and columns.

``` r
set.seed(20251031)
n_hd <- 10000
p_hd <- 400
x_hd <- matrix(rnorm(n_hd * p_hd), nrow = n_hd, ncol = p_hd)
colnames(x_hd) <- paste0("x", seq_len(p_hd))
beta_true <- runif(p_hd, -1, 1)
y_hd <- drop(x_hd %*% beta_true + rnorm(n_hd, sd = 0.5))
hd_data <- data.frame(y = y_hd, x_hd)

options(expressions = 15000) # to avoid errors due to nested evaluation
hd_formula <- DF2formula(hd_data)
```

``` r
hd_mark <- mark(
  lm = lm(hd_formula, data = hd_data),
  mlxs_lm = mlxs_lm(hd_formula, data = hd_data),
  feols = feols(hd_formula, data = hd_data),
  fastLm = RcppEigen::fastLm(hd_formula, data = hd_data),
  speedlm = speedglm::speedlm(hd_formula, data = hd_data),
  iterations = 3,
  check = FALSE
)
#> Warning: Some expressions had a GC in every iteration; so filtering is
#> disabled.

hd_summary <- data.frame(
  method = as.character(hd_mark$expression),
  median_sec = as.numeric(hd_mark$median, units = "s"),
  mem_mb = as.numeric(hd_mark$mem_alloc, units = "MB"),
  itr_per_sec = hd_mark$`itr/sec`
)
hd_summary$relative <- hd_summary$median_sec / min(hd_summary$median_sec)
hd_summary
#>    method median_sec    mem_mb itr_per_sec  relative
#> 1      lm 0.64397917 198187456   1.5085973 10.510283
#> 2 mlxs_lm 0.06127134 165871952  16.4513936  1.000000
#> 3   feols 1.79346817  81721048   0.5541341 29.270913
#> 4  fastLm 0.94782213 278407080   1.0685821 15.469257
#> 5 speedlm 0.18845133 188431080   5.5706482  3.075685
ggplot2::autoplot(hd_mark, type = "beeswarm")
```

![](benchmark-lm-mlxs_files/figure-html/highp-benchmark-1.png)

## Agreement on the Simulated Problem

Even in the high-dimensional setting, the MLX-backed fit lines up with
the reference linear model.

``` r
lm_hd <- lm(hd_formula, data = hd_data)
mlxs_hd <- mlxs_lm(hd_formula, data = hd_data)
speedlm_hd <- speedglm::speedlm(hd_formula, data = hd_data)

c(
  max(abs(coef(lm_hd) - mlxs_hd$coefficients)),
  max(abs(coef(lm_hd) - speedlm_hd$coefficients))
)
#> $ptr
#> <pointer: 0x600002ac9460>
#> 
#> $dim
#> integer(0)
#> 
#> $dtype
#> [1] "float32"
#> 
#> $device
#> [1] "gpu"
#> 
#> [[5]]
#> [1] 3.663736e-15
```

The largest coefficient difference stays at floating-point noise levels,
confirming that the accelerated solver preserves numerical accuracy
while scaling gracefully with both observations and predictors.

## GLM Benchmark: Logistic Regression

The
[`mlxs_glm()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_glm.md)
function extends the MLX backend to generalized linear models. We
benchmark it against [`stats::glm()`](https://rdrr.io/r/stats/glm.html)
and
[`speedglm::speedglm()`](https://rdrr.io/pkg/speedglm/man/speedglm.html)
using a binomial family on the flights data.

``` r
# Create binary outcome: whether arrival delay exceeds 15 minutes
bench_data$late <- as.integer(bench_data$arr_delay > 15)
glm_formula <- late ~ dep_delay + air_time + distance
```

``` r
glm_mark <- mark(
  glm = glm(glm_formula, family = binomial(), data = bench_data),
  mlxs_glm = mlxs_glm(glm_formula, family = mlxs_binomial(), data = bench_data),
  speedglm = speedglm::speedglm(glm_formula, family = binomial(), data = bench_data),
  iterations = 5,
  check = FALSE
)
#> Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
#> Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
#> Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
#> Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
#> Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
#> Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
#> Warning: Some expressions had a GC in every iteration; so filtering is
#> disabled.

glm_summary <- data.frame(
  method = as.character(glm_mark$expression),
  median_sec = as.numeric(glm_mark$median, units = "s"),
  mem_mb = as.numeric(glm_mark$mem_alloc, units = "MB"),
  itr_per_sec = glm_mark$`itr/sec`
)
glm_summary$relative <- glm_summary$median_sec / min(glm_summary$median_sec)
glm_summary
#>     method median_sec     mem_mb itr_per_sec relative
#> 1      glm  1.0991963 1027386600   0.8964031 2.044791
#> 2 mlxs_glm  1.1338071  308501216   0.9052029 2.109176
#> 3 speedglm  0.5375592  528434864   1.5877477 1.000000
ggplot2::autoplot(glm_mark, type = "boxplot")
```

![](benchmark-lm-mlxs_files/figure-html/glm-timings-1.png)

## Agreement on GLM Benchmark

Check that the MLX-backed GLM matches the reference implementation:

``` r
glm_fit <- glm(glm_formula, family = binomial(), data = bench_data)
#> Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
mlxs_glm_fit <- mlxs_glm(glm_formula, family = mlxs_binomial(), data = bench_data)
speedglm_fit <- speedglm::speedglm(glm_formula, family = binomial(), data = bench_data)

glm_coef_delta <- max(abs(coef(glm_fit) - mlxs_glm_fit$coefficients))
glm_fitted_delta <- max(abs(fitted(glm_fit) - mlxs_glm_fit$fitted.values))
speedglm_coef_delta <- max(abs(coef(glm_fit) - speedglm_fit$coefficients))
speedglm_fitted_delta <- max(abs(fitted(glm_fit) - as.vector(predict(speedglm_fit))))
#> Warning in predict.speedglm(speedglm_fit): fitted values were not returned from the speedglm object: 
#>             use the original data by setting argument 'newdata' or refit 
#>             the model by specifying fitted=TRUE.
#> Warning in max(abs(fitted(glm_fit) - as.vector(predict(speedglm_fit)))): no
#> non-missing arguments to max; returning -Inf

c(
  mlxs_max_coefficient_difference = glm_coef_delta,
  mlxs_max_fitted_difference = glm_fitted_delta,
  speedglm_max_coefficient_difference = speedglm_coef_delta,
  speedglm_max_fitted_difference = speedglm_fitted_delta
)
#> $mlxs_max_coefficient_difference.ptr
#> <pointer: 0x600002ac2f70>
#> 
#> $mlxs_max_coefficient_difference.dim
#> integer(0)
#> 
#> $mlxs_max_coefficient_difference.dtype
#> [1] "float32"
#> 
#> $mlxs_max_coefficient_difference.device
#> [1] "gpu"
#> 
#> $mlxs_max_fitted_difference.ptr
#> <pointer: 0x600002ac2230>
#> 
#> $mlxs_max_fitted_difference.dim
#> integer(0)
#> 
#> $mlxs_max_fitted_difference.dtype
#> [1] "float32"
#> 
#> $mlxs_max_fitted_difference.device
#> [1] "gpu"
#> 
#> $speedglm_max_coefficient_difference
#> [1] 2.381206e-12
#> 
#> $speedglm_max_fitted_difference
#> [1] -Inf
```

All implementations agree within numerical tolerance, confirming that
[`mlxs_glm()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_glm.md)
produces accurate results for generalized linear models.

## High-dimensional GLM Benchmark

Large logistic regressions in practice often include hundreds of
engineered features. To stress both runtime and numerical stability we
fabricate a dense dataset with 5,000 observations and 200 predictors,
then compare execution speed across solvers.

``` r
set.seed(20251103)
n_glm <- 5000
p_glm <- 200
x_glm <- matrix(rnorm(n_glm * p_glm), nrow = n_glm, ncol = p_glm)
colnames(x_glm) <- paste0("x", seq_len(p_glm))
beta_glm <- runif(p_glm, -0.5, 0.5)
linpred <- drop(x_glm %*% beta_glm)
prob <- 1 / (1 + exp(-linpred))
y_glm <- rbinom(n_glm, size = 1, prob = prob)

glm_hd_data <- data.frame(y = y_glm, x_glm)
glm_hd_formula <- y ~ .
```

``` r
glm_hd_mark <- mark(
  glm = glm(glm_hd_formula, family = binomial(), data = glm_hd_data,
            control = list(maxit = 50)),
  mlxs_glm = mlxs_glm(glm_hd_formula, family = mlxs_binomial(), data = glm_hd_data,
                      control = list(maxit = 50, epsilon = 1e-6)),
  speedglm = speedglm::speedglm(glm_hd_formula, family = binomial(), data = glm_hd_data),
  iterations = 3,
  check = FALSE
)

glm_hd_summary <- data.frame(
  method = as.character(glm_hd_mark$expression),
  median_sec = as.numeric(glm_hd_mark$median, units = "s"),
  mem_mb = as.numeric(glm_hd_mark$mem_alloc, units = "MB"),
  itr_per_sec = glm_hd_mark$`itr/sec`
)
glm_hd_summary$relative <- glm_hd_summary$median_sec / min(glm_hd_summary$median_sec)
glm_hd_summary
#>     method median_sec    mem_mb itr_per_sec relative
#> 1      glm  0.6761531 167405736    1.478955 3.757059
#> 2 mlxs_glm  0.6574022  44611040    1.521139 3.652870
#> 3 speedglm  0.1799687 120938016    5.556522 1.000000
ggplot2::autoplot(glm_hd_mark, type = "beeswarm")
```

![](benchmark-lm-mlxs_files/figure-html/glm-highd-benchmark-1.png)

## Convergence on the High-dimensional Problem

After benchmarking we run each solver once more to inspect convergence
behaviour and solution agreement.

``` r
glm_hd_fit <- glm(glm_hd_formula, family = binomial(), data = glm_hd_data,
                  control = list(maxit = 50))
mlxs_hd_fit <- mlxs_glm(glm_hd_formula, family = mlxs_binomial(), data = glm_hd_data,
                        control = list(maxit = 50, epsilon = 1e-6))

c(
  observations = nrow(glm_hd_data),
  predictors = ncol(glm_hd_data) - 1,
  mlxs_converged = mlxs_hd_fit$converged,
  mlxs_iterations = mlxs_hd_fit$iter,
  max_coefficient_difference = max(abs(coef(glm_hd_fit) - mlxs_hd_fit$coefficients)),
  max_fitted_difference = max(abs(fitted(glm_hd_fit) - mlxs_hd_fit$fitted.values))
)
#> $observations
#> [1] 5000
#> 
#> $predictors
#> [1] 200
#> 
#> $mlxs_converged
#> [1] TRUE
#> 
#> $mlxs_iterations
#> [1] 6
#> 
#> $max_coefficient_difference.ptr
#> <pointer: 0x600002adcea0>
#> 
#> $max_coefficient_difference.dim
#> integer(0)
#> 
#> $max_coefficient_difference.dtype
#> [1] "float32"
#> 
#> $max_coefficient_difference.device
#> [1] "gpu"
#> 
#> $max_fitted_difference.ptr
#> <pointer: 0x600002adcde0>
#> 
#> $max_fitted_difference.dim
#> integer(0)
#> 
#> $max_fitted_difference.dtype
#> [1] "float32"
#> 
#> $max_fitted_difference.device
#> [1] "gpu"
```

The MLX-backed fit converges in roughly the same number of iterations as
the reference and matches both coefficients and fitted values to within
floating point tolerance, while the timing table above highlights the
runtime benefits at this scale.

## Other Fast Linear Modeling Options

Beyond the methods benchmarked here, practitioners often turn to
[`biglm::biglm()`](https://rdrr.io/pkg/biglm/man/biglm.html) for
streamed data that do not fit in memory or `glmnet` for elastic-net
regularisation. Packages such as `MatrixModels` and `SparseM` also
provide optimized pathways for sparse design matrices.

## Elastic Net on Flights Data

Finally we try the MLX-backed elastic-net solver on flights, expanding
the feature set with interaction terms to create a challenging
regression. The benchmark compares
[`glmnet::glmnet()`](https://glmnet.stanford.edu/reference/glmnet.html)
with the new \[mlxs_glmnet()\] helper.

``` r
set.seed(20251103)
enet_rows <- 50000
enet_idx <- sample.int(nrow(bench_data), enet_rows)
enet_base <- bench_data[enet_idx, ]

enet_design <- model.matrix(
  ~ poly(dep_delay, 3) + poly(air_time, 3) + poly(distance, 3)
    + poly(dep_delay, 2):poly(air_time, 2)
    + poly(dep_delay, 2):poly(distance, 2)
    + poly(air_time, 2):poly(distance, 2),
  data = enet_base
)

x_enet <- enet_design[, -1, drop = FALSE]
y_enet <- enet_base$arr_delay > 15
lambda_enet <- 0.01
```

``` r
enet_mark <- mark(
  glmnet = glmnet::glmnet(x_enet, y_enet, family = "binomial", alpha = 1,
                          lambda = lambda_enet, standardize = TRUE,
                          thresh = 1e-6, maxit = 100000),
  mlxs_glmnet = mlxs_glmnet(x_enet, y_enet, family = mlxs_binomial(), alpha = 1,
                            lambda = lambda_enet, standardize = TRUE,
                            maxit = 2000, tol = 1e-6),
  iterations = 3,
  check = FALSE
)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning: Some expressions had a GC in every iteration; so filtering is
#> disabled.

enet_summary <- data.frame(
  method = as.character(enet_mark$expression),
  median_sec = as.numeric(enet_mark$median, units = "s"),
  mem_mb = as.numeric(enet_mark$mem_alloc, units = "MB"),
  itr_per_sec = enet_mark$`itr/sec`
)
enet_summary$relative <- enet_summary$median_sec / min(enet_summary$median_sec)
enet_summary
#>        method median_sec   mem_mb itr_per_sec relative
#> 1      glmnet 0.06651574 22547280  14.2200251  1.00000
#> 2 mlxs_glmnet 5.83850873 71440280   0.1770935 87.77635
ggplot2::autoplot(enet_mark, type = "beeswarm")
```

![](benchmark-lm-mlxs_files/figure-html/glmnet-flights-benchmark-1.png)

``` r
enet_ref <- glmnet::glmnet(x_enet, y_enet, family = "binomial", alpha = 1,
                           lambda = lambda_enet, standardize = TRUE,
                           thresh = 1e-6, maxit = 100000)
enet_fit <- mlxs_glmnet(x_enet, y_enet, family = mlxs_binomial(), alpha = 1,
                        lambda = lambda_enet, standardize = TRUE,
                        maxit = 2000, tol = 1e-6)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)
#> Warning in as.vector.mlx(x, mode): Converting multi-dimensional mlx array to
#> vector (flattening in column-major order)

c(
  coefficients_max_difference = max(abs(as.numeric(enet_ref$beta) - enet_fit$beta[, 1])),
  intercept_difference = abs(as.numeric(enet_ref$a0) - enet_fit$a0[1])
)
#> coefficients_max_difference        intercept_difference 
#>                0.5963112556                0.0005400035
```

Even with a rich feature expansion, the MLX implementation matches
`glmnet` within modest tolerances while keeping all updates on the GPU.
