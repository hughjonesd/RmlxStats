# MLX-backed generalized linear model

Fit generalized linear models using iterative reweighted least squares
(IRLS) with MLX providing the heavy lifting for weighted least squares
solves. Final convergence is done at double precision on the cpu.

## Usage

``` r
mlxs_glm(
  formula,
  family = mlxs_gaussian(),
  data,
  subset,
  weights,
  na.action = stats::na.exclude,
  start = NULL,
  control = list(),
  ...
)
```

## Arguments

- formula:

  Model formula.

- family:

  A mlxs family object (e.g.,
  [`mlxs_gaussian()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_gaussian.md),
  [`mlxs_binomial()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_binomial.md),
  [`mlxs_poisson()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_poisson.md)).
  You can use `"gaussian"` etc.

- data:

  Optional data frame, tibble, or environment containing the variables
  in the model.

- subset:

  Optional expression for subsetting observations.

- weights:

  Optional non-negative observation weights.

- na.action:

  How to handle missing values.

- start:

  Starting values for the parameters in the linear predictor.

- control:

  Optional list of control parameters passed to
  [`mlxs_glm_control()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_glm_control.md).
  Control parameters can include `epsilon`, `epsilon_f64`, `maxit`,
  `trace`, and `rank_tol`.

- ...:

  Additional arguments passed to the family function when `family` is
  supplied as a function or string.

## Value

An object of class `c("mlxs_glm", "mlxs_model")` containing elements
similar to the result of
[`stats::glm()`](https://rdrr.io/r/stats/glm.html). Unlike
[`stats::glm()`](https://rdrr.io/r/stats/glm.html), rank-deficient model
matrices are rejected rather than fit with aliased coefficients.

## Examples

``` r
fit <- mlxs_glm(mpg ~ cyl + disp, family = mlxs_gaussian(), data = mtcars)
coef(fit)
#> (Intercept)         cyl        disp 
#> 34.66099167 -1.58727658 -0.02058364 
```
