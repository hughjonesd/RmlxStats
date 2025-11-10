# MLX-backed linear regression

Fit a linear model via QR decomposition using MLX arrays on Apple
Silicon devices. The interface mirrors
[`stats::lm()`](https://rdrr.io/r/stats/lm.html) for the common
arguments.

## Usage

``` r
mlxs_lm(formula, data, subset)
```

## Arguments

- formula:

  Model formula.

- data:

  Optional data frame, tibble, or environment containing the variables
  in the model.

- subset:

  Optional expression for subsetting observations.

## Value

An object of class `c("mlxs_lm", "mlxs_model")` containing components
similar to an `"lm"` fit, along with MLX intermediates stored in the
`mlx` element. Note that MLX currently operates in single precision, so
fitted values and diagnostics may differ from
[`stats::lm()`](https://rdrr.io/r/stats/lm.html) at around the 1e-6
level.

## Examples

``` r
fit <- mlxs_lm(mpg ~ cyl + disp, data = mtcars)
coef(fit)
#> mlx array [3 x 1]
#>   dtype: float32
#>   device: gpu
#>   values:
#>             [,1]
#> [1,] 34.66099167
#> [2,] -1.58727658
#> [3,] -0.02058364
#> attr(,"coef_names")
#> [1] "(Intercept)" "cyl"         "disp"       
```
