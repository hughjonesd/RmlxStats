# MLX-backed linear regression

Fit a linear model via QR decomposition using MLX arrays on Apple
Silicon devices. The interface mirrors
[`stats::lm()`](https://rdrr.io/r/stats/lm.html) for the common
arguments.

## Usage

``` r
mlxs_lm(
  formula,
  data,
  subset,
  weights,
  na.action = stats::na.exclude,
  rank_tol = NULL
)
```

## Arguments

- formula:

  Model formula.

- data:

  Optional data frame, tibble, or environment containing the variables
  in the model.

- subset:

  Optional expression for subsetting observations.

- weights:

  Optional non-negative observation weights.

- na.action:

  How to handle missing values.

- rank_tol:

  Optional relative tolerance used to detect rank-deficient systems.
  `NULL` uses the package default, which varies by dtype and is 1e-6 for
  float32 matrices. Set to `FALSE` to skip rank checks entirely. Note
  that higher numbers indicate *lower* tolerance.

## Value

An object of class `c("mlxs_lm", "mlxs_model")` containing components
similar to an `"lm"` fit, along with MLX intermediates stored in the
`mlx` element. Note that MLX currently operates in single precision, so
fitted values and diagnostics may differ from
[`stats::lm()`](https://rdrr.io/r/stats/lm.html) at around the 1e-6
level. Unlike [`stats::lm()`](https://rdrr.io/r/stats/lm.html),
rank-deficient model matrices are rejected rather than fit with aliased
coefficients.

## Examples

``` r
fit <- mlxs_lm(mpg ~ cyl + disp, data = mtcars)
coef(fit)
#> (Intercept)         cyl        disp 
#> 34.66099167 -1.58727658 -0.02058364 
```
