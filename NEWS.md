# RmlxStats 0.4.0

* `mlxs_lm_fit()` now uses Rmlx's specialized code for QR on the GPU when the
  design has more than 10 million elements, with `qr_method` available for
  explicit selection. 
  
  
# RmlxStats 0.3.0

* `mlxs_glm()` now moves to float64 on the cpu where necessary to 
  compute more accurate estimates.
* New `mlxs_glm_control()` function.
* `mlxs_lm()` and `mlxs_glm()` now reject rank-deficient `x`. A bug which 
  meant we calculated `qr(x)` twice has now been fixed.
* `mlxs_lm()`, `mlxs_lm_fit()`, and `mlxs_glm_control()` now expose
  `rank_tol` to tune rank-deficiency detection. Set `rank_tol = FALSE`
  to skip rank checks entirely.
* Bugfix: `mlxs_lm()` now drops unused factor levels.
* New `bread()`, `estfun()` and `hatvalues()` methods for `mlxs_lm` to allow for
  sandwich-style robust standard errors. 
* More `mlxs_lm` methods now return base R objects by default, controllable
  via the `output` argument.
* `confint.mlxs_lm()` and `confint.mlxs_glm()` can now return bootstrap
  confidence intervals. So can the respective `summary()` methods.
* Speedups for some `augment()`, `predict()` and `summary()` methods.


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
