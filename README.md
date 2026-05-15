# RmlxStats

Run statistics on your Mac GPU  with [mlx](https://mlx-framework.org) and
[Rmlx](https://github.com/hughjonesd/Rmlx).

GPUs are designed to handle matrices, which is a good fit for statistics.
But up till now R Mac users have not had access to the power of their GPUs.
RmlxStats is an experiment in implementing common statistical methods on the 
GPU. RmlxStats is early *work in progress*!

Functions implemented so far include Rmlx versions of `lm`, `glm`, `glmnet` and
`prcomp`, and `mlxs_boot()` for bootstrapping.

## When to use

RmlxStats offers large speedups against both base R functions, and 
speed-optimized packages like speedglm and RCppEigen. Speedups are especially 
large for regressions with many predictors (large p). Very roughly, if you
have 50 or more predictors and 10,000 or more rows, or if your regressions
are taking measurable time to complete, RmlxStats is worth trying. See the 
benchmarks vignette for more details.

GPU calculations use float32 precision, so if you need higher numerical accuracy
than this, RmlxStats may not be the right tool (though `mlxs_glm()` can
now finish fitting in float64 on the CPU).

## Installation

Install Apple's MLX runtime:

   ```bash
   brew install mlx
   ```

Then:

   ```r
   remotes::install_github("hughjonesd/RmlxStats")
   ```

which will also install Rmlx.