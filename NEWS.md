# RmlxStats (development version)

# RmlxStats 0.2.0

* Added `mlxs_prcomp()`, a `prcomp()`-style PCA interface with exact and
  randomized truncated MLX-backed decomposition paths. Benchmarks show
  this greatly outperforms base R `prcomp()` and other specialised packages
  for fast PCA.
* Reworked `mlxs_glmnet()`. It can now outperform `glmnet::glmnet()` for large 
  problems (roughly n x p > 5,000,000).
* Added `mlxs_cv_glmnet()` as a cross-validation wrapper for the MLX-backed
  elastic-net path fits, analogous to `glmnet::cv.glmnet()`.
* Export `mlxs_lm_fit()` so advanced users can call the MLX-backed QR solver directly.

# RmlxStats 0.1.0

* Initial version.
