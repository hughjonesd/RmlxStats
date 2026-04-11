# RmlxStats (development version)

* Reworked `mlxs_glmnet()` to use dense on-device MLX updates, removing the
  active-set slicing/scatter hotspot from the Gaussian and binomial paths.
* Added a Gaussian covariance-space path for very tall problems, so
  `mlxs_glmnet()` can reuse `X'X / n` when `n` is much larger than `p`.
* Export `mlxs_lm_fit()` so advanced users can call the MLX-backed QR solver directly.

# RmlxStats 0.1.0

* Initial version.
