# PCA methods for `mlxs_prcomp`

`predict.mlxs_prcomp()` returns MLX scores. The presentation methods
([`print()`](https://rdrr.io/r/base/print.html),
[`summary()`](https://rdrr.io/r/base/summary.html),
[`plot()`](https://rdrr.io/r/graphics/plot.default.html), and
[`biplot()`](https://rdrr.io/r/stats/biplot.html)) reuse the base
`prcomp` implementations by converting to a temporary host-backed
`prcomp` object.

## Usage

``` r
# S3 method for class 'mlxs_prcomp'
predict(object, newdata, ...)

# S3 method for class 'mlxs_prcomp'
print(x, ...)

# S3 method for class 'mlxs_prcomp'
summary(object, ...)

# S3 method for class 'mlxs_prcomp'
plot(x, ...)

# S3 method for class 'mlxs_prcomp'
biplot(x, ...)

# S3 method for class 'mlxs_prcomp'
nobs(object, ...)

# S3 method for class 'mlxs_prcomp'
tidy(x, ...)

# S3 method for class 'mlxs_prcomp'
augment(x, data = NULL, newdata = NULL, output = c("data.frame", "mlx"), ...)
```

## Arguments

- object, x:

  A fitted `mlxs_prcomp` object.

- newdata:

  Optional new observations to project.

- ...:

  Passed through to the corresponding base method.

- data:

  Optional original data to append PCA scores to in
  `augment.mlxs_prcomp()`.

- output:

  Output format for `augment.mlxs_prcomp()`: either a data frame with
  appended score columns or the MLX score matrix directly.

## Value

Method-specific output. `predict.mlxs_prcomp()` returns an MLX matrix.
`augment.mlxs_prcomp()` returns either a data frame or MLX matrix.
