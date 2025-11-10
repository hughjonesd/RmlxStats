# MLX-backed generalized linear model

Fit generalized linear models using iterative reweighted least squares
(IRLS) with MLX providing the heavy lifting for weighted least squares
solves.

## Usage

``` r
mlxs_glm(
  formula,
  family = mlxs_gaussian(),
  data,
  subset,
  na.action,
  start = NULL,
  control = list(),
  ...
)
```

## Arguments

- formula:

  an object of class
  `"`[`formula`](https://rdrr.io/r/stats/formula.html)`"` (or one that
  can be coerced to that class): a symbolic description of the model to
  be fitted. The details of model specification are given under
  ‘Details’.

- family:

  A mlxs family object (e.g.,
  [`mlxs_gaussian()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_gaussian.md),
  [`mlxs_binomial()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_binomial.md),
  [`mlxs_poisson()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_poisson.md)).

- data:

  an optional data frame, list or environment (or object coercible by
  [`as.data.frame`](https://rdrr.io/r/base/as.data.frame.html) to a data
  frame) containing the variables in the model. If not found in `data`,
  the variables are taken from `environment(formula)`, typically the
  environment from which `glm` is called.

- subset:

  an optional vector specifying a subset of observations to be used in
  the fitting process.

- na.action:

  a function which indicates what should happen when the data contain
  `NA`s. The default is set by the `na.action` setting of
  [`options`](https://rdrr.io/r/base/options.html), and is
  [`na.fail`](https://rdrr.io/r/stats/na.fail.html) if that is unset.
  The ‘factory-fresh’ default is
  [`na.omit`](https://rdrr.io/r/stats/na.fail.html). Another possible
  value is `NULL`, no action. Value
  [`na.exclude`](https://rdrr.io/r/stats/na.fail.html) can be useful.

- start:

  starting values for the parameters in the linear predictor.

- control:

  Optional list of control parameters passed to
  [`stats::glm.control()`](https://rdrr.io/r/stats/glm.control.html).

- ...:

  For `glm`: arguments to be used to form the default `control` argument
  if it is not supplied directly.

  For `weights`: further arguments passed to or from other methods.

## Value

An object of class `c("mlxs_glm", "mlxs_model")` containing elements
similar to the result of
[`stats::glm()`](https://rdrr.io/r/stats/glm.html). MLX intermediates
are stored in the `mlx` field for downstream reuse. Computations use
single-precision MLX tensors, so results typically agree with
[`stats::glm()`](https://rdrr.io/r/stats/glm.html) to around 1e-6 unless
a tighter tolerance is supplied via `control`.

## Examples

``` r
fit <- mlxs_glm(mpg ~ cyl + disp, family = mlxs_gaussian(), data = mtcars)
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
