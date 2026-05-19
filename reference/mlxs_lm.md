# MLX-backed linear regression

Fit a linear model via QR decomposition using MLX arrays on Apple
Silicon devices. The interface mirrors
[`stats::lm()`](https://rdrr.io/r/stats/lm.html) for the common
arguments.

## Usage

``` r
mlxs_lm(formula, data, subset, weights, na.action = stats::na.exclude)
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

  Optional non-negative observation weights. Treated like the `weights`
  argument to [`stats::lm()`](https://rdrr.io/r/stats/lm.html), i.e.
  they enter the fit via weighted least squares.

- na.action:

  A function indicating how missing values should be handled. Defaults
  to [`stats::na.exclude()`](https://rdrr.io/r/stats/na.fail.html) so
  residuals, fitted values, and training-data predictions are padded
  back to the original row count.

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
