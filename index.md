# RmlxStats

Statistical modelling front-ends that run on Apple GPU hardware via the
[Rmlx](https://github.com/hughjonesd/Rmlx) array library.

GPUs are designed to handle matrices, which is a good fit for
statistics. But up till now R Mac users have not had access to the power
of their GPUs. RmlxStats is an experiment in implementing common
statistical methods on the GPU. RmlxStats is early *work in progress*!

Functions implemented so far include Rmlx versions of `lm`, `glm`,
`glmnet` and a bootstrapping function
[`mlxs_boot()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_boot.md).

## When to use

Very roughly, RmlxStats becomes competitive once you have ~10,000 rows
and ~50 columns. Below that, startup costs dominate. See the benchmarks
vignette for more details.

## Installation

Install Apple’s MLX runtime (provides the Metal-backed tensor engine):

``` bash
brew install mlx
```

Then:

``` r
remotes::install_github("hughjonesd/RmlxStats")
```

which will also install Rmlx.
