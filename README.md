# RmlxStats

Statistical modelling front-ends that run on Apple GPU hardware 
via the [Rmlx](https://github.com/hughjonesd/Rmlx) array library.

GPUs are designed to handle matrices, which is a good fit for statistics.
But up till now R Mac users have not had access to the power of their GPUs.
RmlxStats is an experiment in implementing common statistical methods on the 
GPU. RmlxStats is early *work in progress*!

Functions implemented so far include Rmlx versions of `lm`, `glm`, `glmnet`
and a bootstrapping function `mlxs_boot()`.

## When to use

RmlxStats offers large speedups against both base R functions, and 
speed-optimized packages like speedglm and RCppEigen. Speedups are especially 
large for regressions with many predictors (large p). 

Very roughly, if you
have 50 or more predictors and 10,000 or more rows, or if your regressions
are taking measurable time to complete, RmlxStats is worth trying:

  ```r
  # On my machine
  > system.time({lm <- lm(arr_delay ~ dep_delay + factor(paste(month,day)), data = nycflights13::flights); })
     user  system elapsed 
   31.310   0.179  31.479 

  > system.time({lm2 <- mlxs_lm(arr_delay ~ dep_delay + factor(paste(month,day)), data = nycflights13::flights); Rmlx::mlx_eval(lm2$coefficients)})
     user  system elapsed 
    4.421   0.271   2.818 
  ```

See the benchmarks vignette for more details.

GPU calculations use float32 precision, so if you need higher numerical accuracy
than this, RmlxStats may not be the right tool.

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