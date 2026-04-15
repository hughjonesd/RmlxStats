# RmlxStats (development version)

* Reworked `mlxs_glmnet()` to use dense on-device MLX updates, removing the
  active-set slicing/scatter hotspot from the Gaussian and binomial paths.
* Added a Gaussian covariance-space path for very tall problems, so
  `mlxs_glmnet()` can reuse `X'X / n` when `n` is much larger than `p`.
* Cache `mlx_compile()`'d chunk updates inside `mlxs_glmnet()`, cutting the
  host overhead for Gaussian and binomial path iterations.
* Added `mlxs_cv_glmnet()` as a cross-validation wrapper for the MLX-backed
  elastic-net path fits, along with `coef()` and `predict()` methods for
  `mlxs_glmnet` and `mlxs_cv_glmnet`.
* Export `mlxs_lm_fit()` so advanced users can call the MLX-backed QR solver directly.

# RmlxStats 0.1.0

* Initial version.
