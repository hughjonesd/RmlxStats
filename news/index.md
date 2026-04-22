# Changelog

## RmlxStats 0.2.0

- Added
  [`mlxs_prcomp()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_prcomp.md),
  a [`prcomp()`](https://rdrr.io/r/stats/prcomp.html)-style PCA
  interface with exact and randomized truncated MLX-backed decomposition
  paths.
- Reworked
  [`mlxs_glmnet()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_glmnet.md)
  to use dense on-device MLX updates, removing the active-set
  slicing/scatter hotspot from the Gaussian and binomial paths.
- Added a Gaussian covariance-space path for very tall problems, so
  [`mlxs_glmnet()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_glmnet.md)
  can reuse `X'X / n` when `n` is much larger than `p`.
- [`mlxs_glmnet()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_glmnet.md)
  can now outperform
  [`glmnet::glmnet()`](https://glmnet.stanford.edu/reference/glmnet.html)
  for large problems (roughly n x p \> 5,000,000).
- Cache `mlx_compile()`’d chunk updates inside
  [`mlxs_glmnet()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_glmnet.md),
  cutting the host overhead for Gaussian and binomial path iterations.
- Added
  [`mlxs_cv_glmnet()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_cv_glmnet.md)
  as a cross-validation wrapper for the MLX-backed elastic-net path
  fits, along with [`coef()`](https://rdrr.io/r/stats/coef.html) and
  [`predict()`](https://rdrr.io/r/stats/predict.html) methods for
  `mlxs_glmnet` and `mlxs_cv_glmnet`.
- Export
  [`mlxs_lm_fit()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_lm_fit.md)
  so advanced users can call the MLX-backed QR solver directly.

## RmlxStats 0.1.0

- Initial version.
