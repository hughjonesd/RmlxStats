# Changelog

## RmlxStats (development version)

- [`mlxs_glm()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_glm.md)
  now moves to float64 on the cpu where necessary to compute more
  accurate estimates.
- New
  [`mlxs_glm_control()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_glm_control.md)
  function.
- [`mlxs_lm()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_lm.md)
  and
  [`mlxs_glm()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_glm.md)
  now reject rank-deficient `x`. A bug which meant we calculated `qr(x)`
  twice has now been fixed.
- New `bread()`, `estfun()` and
  [`hatvalues()`](https://rdrr.io/r/stats/influence.measures.html)
  methods for `mlxs_lm` to allow for sandwich-style robust standard
  errors.
- More `mlxs_lm` methods now return base R objects by default,
  controllable via the `output` argument.
- [`confint.mlxs_lm()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs-lm-methods.md)
  and
  [`confint.mlxs_glm()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs-glm-methods.md)
  can now return bootstrap confidence intervals. So can the respective
  [`summary()`](https://rdrr.io/r/base/summary.html) methods.
- Speedups for some
  [`augment()`](https://hughjonesd.github.io/RmlxStats/reference/generics-reexports.md),
  [`predict()`](https://rdrr.io/r/stats/predict.html) and
  [`summary()`](https://rdrr.io/r/base/summary.html) methods.

## RmlxStats 0.2.0

- Added
  [`mlxs_prcomp()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_prcomp.md),
  a [`prcomp()`](https://rdrr.io/r/stats/prcomp.html)-style PCA
  interface with exact and randomized truncated MLX-backed decomposition
  paths. Benchmarks show this greatly outperforms base R
  [`prcomp()`](https://rdrr.io/r/stats/prcomp.html) and other
  specialised packages for fast PCA.
- Reworked
  [`mlxs_glmnet()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_glmnet.md).
  It can now outperform
  [`glmnet::glmnet()`](https://glmnet.stanford.edu/reference/glmnet.html)
  for large problems (roughly n x p \> 5,000,000).
- Added
  [`mlxs_cv_glmnet()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_cv_glmnet.md)
  as a cross-validation wrapper for the MLX-backed elastic-net path
  fits, analogous to
  [`glmnet::cv.glmnet()`](https://glmnet.stanford.edu/reference/cv.glmnet.html).
- Export
  [`mlxs_lm_fit()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_lm_fit.md)
  so advanced users can call the MLX-backed QR solver directly.

## RmlxStats 0.1.0

- Initial version.
