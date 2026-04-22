# MLX-backed elastic net regression

Fit lasso or elastic-net penalised regression paths using MLX arrays for
the heavy linear algebra. Dense Gaussian and binomial paths stay on the
MLX backend throughout the iterative updates, with repeated chunk
updates traced through
[`Rmlx::mlx_compile()`](https://hughjonesd.github.io/Rmlx/reference/mlx_compile.html)
to reduce host overhead.

## Usage

``` r
mlxs_glmnet(
  x,
  y,
  family = mlxs_gaussian(),
  alpha = 1,
  lambda = NULL,
  nlambda = 100,
  lambda_min_ratio = 1e-04,
  standardize = TRUE,
  intercept = TRUE,
  use_strong_rules = TRUE,
  maxit = 1000,
  tol = 1e-06
)
```

## Arguments

- x:

  Numeric matrix of predictors (observations in rows).

- y:

  Numeric response vector (for binomial, values must be 0/1).

- family:

  MLX-aware family object, e.g.
  [`mlxs_gaussian()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_gaussian.md)
  or
  [`mlxs_binomial()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_binomial.md).

- alpha:

  Elastic-net mixing parameter (1 = lasso, currently alpha must be
  strictly positive).

- lambda:

  Optional decreasing sequence of penalty values. If `NULL`, a sequence
  of length `nlambda` is generated from `lambda_max` down to
  `lambda_max * lambda_min_ratio`.

- nlambda:

  Length of automatically generated lambda path.

- lambda_min_ratio:

  Smallest lambda as a fraction of `lambda_max`.

- standardize:

  Should columns of `x` be scaled before fitting?

- intercept:

  Should an intercept be fit?

- use_strong_rules:

  Retained for API compatibility. The dense MLX solver keeps all
  coefficients on device, so this flag currently does not change the
  computation.

- maxit:

  Maximum proximal-gradient iterations per lambda value.

- tol:

  Convergence tolerance on the coefficient updates.

## Value

An object of class `mlxs_glmnet` containing the fitted coefficient path,
intercepts, lambda sequence, and scaling information.

## Note

[`glmnet::glmnet()`](https://glmnet.stanford.edu/reference/glmnet.html)
is faster on smaller problems. Very roughly as of April 2026,
`mlxs_glmnet()` gets competitive at n x p = 5,000,000 or greater.
