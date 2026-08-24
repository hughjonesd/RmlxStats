# Fit an MLX linear model from design matrices

`mlxs_lm_fit()` powers
[`mlxs_lm()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_lm.md)
by wrapping the QR-based solver that runs entirely on MLX arrays.

## Usage

``` r
mlxs_lm_fit(
  x,
  y,
  weights = NULL,
  rank_tol = NULL,
  qr_method = c("auto", "cpu", "cholqr", "tsqr")
)
```

## Arguments

- x:

  MLX design matrix (or object coercible via
  [`Rmlx::as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.html))
  whose rows represent observations and columns represent predictors.

- y:

  MLX column vector (or object coercible via
  [`Rmlx::as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.html))
  holding the response values.

- weights:

  Optional MLX column vector or numeric vector of non-negative
  observation weights. When supplied, weighted least squares are fit via
  the standard square-root weighting.

- rank_tol:

  Optional relative tolerance used to detect rank-deficient systems.
  `NULL` uses the package default, which varies by dtype and is 1e-6 for
  float32 matrices. Set to `FALSE` to skip rank checks entirely. Note
  that higher numbers indicate *lower* tolerance.

- qr_method:

  QR implementation. `"auto"` uses Rmlx Cholesky QR on the GPU when
  `nrow(x) * ncol(x) > 1e7` and otherwise uses MLX QR on the CPU.
  `"cholqr"`, `"tsqr"`, and `"cpu"` force a specific implementation.

## Value

A list with components `coefficients`, `fitted.values`, `residuals`,
`effects`, and `qr`, mirroring the corresponding pieces of
[`stats::lm()`](https://rdrr.io/r/stats/lm.html). Array-valued
components remain MLX matrices to keep downstream GPU pipelines in
device memory.

## Details

Inputs that are not already MLX objects are converted with
[`Rmlx::as_mlx()`](https://hughjonesd.github.io/Rmlx/reference/as_mlx.html)
or
[`Rmlx::mlx_matrix()`](https://hughjonesd.github.io/Rmlx/reference/mlx_matrix.html)
so callers can provide base-R matrices or vectors. Weighted fits are
performed by applying the standard square-root weight transform before
solving the QR system. Rmlx applies a GPU residual-correction pass to
well-conditioned Cholesky QR fits.

## Examples

``` r
x <- Rmlx::as_mlx(cbind(1, as.matrix(mtcars[c("cyl", "disp")])))
y <- Rmlx::mlx_matrix(mtcars$mpg, ncol = 1)
fit <- mlxs_lm_fit(x, y)
drop(as.matrix(fit$coefficients))
#>                     cyl        disp 
#> 34.66099167 -1.58727658 -0.02058364 
```
