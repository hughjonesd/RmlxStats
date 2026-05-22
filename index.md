# RmlxStats

Statistical modelling front-ends that run on Apple GPU hardware via the
[Rmlx](https://github.com/hughjonesd/Rmlx) array library.

GPUs are designed to handle matrices, which is a good fit for
statistics. But up till now R Mac users have not had access to the power
of their GPUs. RmlxStats is an experiment in implementing common
statistical methods on the GPU. RmlxStats is early *work in progress*!

Functions implemented so far include Rmlx versions of `lm`, `glm`,
`glmnet` and `prcomp`, and
[`mlxs_boot()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_boot.md)
for bootstrapping.

## When to use

RmlxStats offers large speedups against both base R functions, and
speed-optimized packages like speedglm and RCppEigen. Speedups are
especially large for regressions with many predictors (large p).

Very roughly, if you have 50 or more predictors and 10,000 or more rows,
or if your regressions are taking measurable time to complete, RmlxStats
is worth trying:

``` r
# On my machine
> system.time(lm(arr_delay ~ dep_delay + factor(paste(month,day)), 
                 data = nycflights13::flights))
   user  system elapsed 
 31.769   0.544  32.764 

> system.time({
    fit <- mlxs_lm(arr_delay ~ dep_delay + factor(paste(month,day)), 
                   data = nycflights13::flights)
    Rmlx::mlx_eval(fit$coefficients)
  })
   user  system elapsed 
  4.274   0.739   3.351 
```

See the benchmarks vignette for more details.

GPU calculations use float32 precision, so if you need higher numerical
accuracy than this, RmlxStats may not be the right tool (though
[`mlxs_glm()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_glm.md)
can now finish in 64-bit precision on the CPU).

## Installation

Install Apple’s MLX runtime:

``` bash
brew install mlx
```

Then:

``` r

remotes::install_github("hughjonesd/RmlxStats")
```

which will also install Rmlx.
