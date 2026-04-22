# Cross-validated MLX elastic net regression

Cross-validation wrapper around
[`mlxs_glmnet()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_glmnet.md)
that mirrors the core
[`glmnet::cv.glmnet()`](https://glmnet.stanford.edu/reference/cv.glmnet.html)
workflow for the families currently supported by
[`mlxs_glmnet()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_glmnet.md).

## Usage

``` r
mlxs_cv_glmnet(
  x,
  y,
  weights = NULL,
  offset = NULL,
  lambda = NULL,
  type.measure = c("default", "mse", "deviance", "class", "mae", "auc", "C"),
  nfolds = 10,
  foldid = NULL,
  alignment = c("lambda", "fraction"),
  grouped = TRUE,
  keep = FALSE,
  parallel = FALSE,
  gamma = c(0, 0.25, 0.5, 0.75, 1),
  relax = FALSE,
  trace.it = 0,
  family = mlxs_gaussian(),
  ...
)
```

## Arguments

- x:

  Numeric matrix of predictors (observations in rows).

- y:

  Numeric response vector.

- weights:

  Optional observation weights. Currently unsupported.

- offset:

  Optional offset. Currently unsupported.

- lambda:

  Optional decreasing lambda sequence. If `NULL`, the full-data fit
  chooses the path and the same path is reused inside each fold.

- type.measure:

  Loss used to score the holdout predictions.

- nfolds:

  Number of folds.

- foldid:

  Optional integer vector giving the fold assignment for each
  observation.

- alignment:

  Alignment mode. Only `"lambda"` is currently supported.

- grouped:

  Should cross-validation be aggregated fold-by-fold? Only `TRUE` is
  currently supported.

- keep:

  Should out-of-fold predictions be stored?

- parallel:

  Logical. Parallel refits are currently unsupported.

- gamma, relax:

  Relaxed fits are currently unsupported.

- trace.it:

  Progress tracing. Currently unsupported.

- family:

  MLX-aware family object, e.g.
  [`mlxs_gaussian()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_gaussian.md)
  or
  [`mlxs_binomial()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_binomial.md).

- ...:

  Additional arguments passed to
  [`mlxs_glmnet()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_glmnet.md),
  such as `alpha`, `nlambda`, `lambda_min_ratio`, `standardize`,
  `intercept`, `maxit`, and `tol`.

## Value

An object of class `mlxs_cv_glmnet`.

## Details

The full-data fit defines a master lambda path. Each fold is then refit
on the same lambda values and scored on its holdout set.

Current limitations relative to
[`glmnet::cv.glmnet()`](https://glmnet.stanford.edu/reference/cv.glmnet.html):

- only Gaussian and binomial families are supported

- `weights`, `offset`, `alignment != "lambda"`, `grouped = FALSE`,
  `parallel = TRUE`, `relax = TRUE`, and non-zero `trace.it` are not
  implemented

- `type.measure = "auc"` and `type.measure = "C"` are not implemented
