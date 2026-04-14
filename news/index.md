# Changelog

## RmlxStats (development version)

- Reworked
  [`mlxs_glmnet()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_glmnet.md)
  to use dense on-device MLX updates, removing the active-set
  slicing/scatter hotspot from the Gaussian and binomial paths.
- Added a Gaussian covariance-space path for very tall problems, so
  [`mlxs_glmnet()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_glmnet.md)
  can reuse `X'X / n` when `n` is much larger than `p`.
- Cache `mlx_compile()`’d chunk updates inside
  [`mlxs_glmnet()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_glmnet.md),
  cutting the host overhead for Gaussian and binomial path iterations.
- Export
  [`mlxs_lm_fit()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_lm_fit.md)
  so advanced users can call the MLX-backed QR solver directly.

## RmlxStats 0.1.0

- Initial version.
