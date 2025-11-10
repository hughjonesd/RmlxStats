# Upstream Issue Backlog (to be filed on GitHub)

1. **anova.mlxs_lm** – implement a genuine MLX-native deviance table so we stop punting back to base `lm` (requested 2025-11-10).
2. **mlxs_quantile / quantile.mlx** – expose percentile helpers built from existing MLX ops so summaries/print methods never have to drop to R just to compute quantiles.
3. **Prediction SEs** – add variance propagation so `predict.mlxs_lm()` / `mlxs_glm()` and `augment(..., se_fit = TRUE)` can supply MLX-resident standard errors.
4. **as.numeric.mlx** – upstream an official converter (likely via `as.numeric(as.vector(x))`) so packages like RmlxStats can stop maintaining bespoke flattening helpers.
