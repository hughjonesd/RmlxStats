# Bootstrap MLX arrays along the first dimension

`mlxs_boot()` resamples observations from one or more MLX arrays, calls
a user-supplied function on each resampled batch, and returns the
collected results. Every argument supplied via `...` must share the same
size in its first dimension (number of observations). Arguments that do
not need resampling should be captured in the environment of `fun`
instead of being passed through `...`.

## Usage

``` r
mlxs_boot(
  fun,
  ...,
  B = 200L,
  seed = NULL,
  progress = FALSE,
  replace = TRUE,
  compile = FALSE
)
```

## Arguments

- fun:

  Function called on each bootstrap draw. It must accept the same named
  arguments as supplied through `...`.

- ...:

  Arrays, matrices, or vectors that should be resampled along the first
  dimension before being passed to `fun`.

- B:

  Number of bootstrap iterations.

- seed:

  Optional integer seed for reproducibility.

- progress:

  Logical; if `TRUE`, show a text progress bar.

- replace:

  Logical; whether to sample with replacement. Defaults to `TRUE` for
  standard bootstrap resampling.

- compile:

  Logical; compile `fun` once via
  [`Rmlx::mlx_compile()`](https://hughjonesd.github.io/Rmlx/reference/mlx_compile.html)
  before entering the resampling loop. Defaults to `FALSE`.

## Value

A list with elements `samples` (the raw results from `fun`), `B`, and
`seed`.
