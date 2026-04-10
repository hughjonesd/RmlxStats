# RmlxStats (development version)

* Reworked `mlxs_glmnet()` to use dense on-device MLX updates, removing the
  active-set slicing/scatter hotspot from the Gaussian and binomial paths.
* Export `mlxs_lm_fit()` so advanced users can call the MLX-backed QR solver directly.

# RmlxStats 0.1.0

* Initial version.
