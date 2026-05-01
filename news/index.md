# Changelog

## RmlxStats (development version)

- [`mlxs_lm()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_lm.md)
  and
  [`mlxs_glm()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_glm.md)
  now reject rank-deficient model matrices with a clear error rather
  than returning unstable aliased coefficients.

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
