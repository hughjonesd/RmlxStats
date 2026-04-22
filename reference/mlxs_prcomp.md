# MLX-backed principal components analysis

Perform principal components analysis with MLX arrays, keeping the
centred and scaled data on device throughout the decomposition.

## Usage

``` r
mlxs_prcomp(
  x,
  retx = TRUE,
  center = TRUE,
  scale. = FALSE,
  tol = NULL,
  rank. = NULL,
  oversample = 10L,
  n_iter = 2L,
  seed = 1L,
  ...
)
```

## Arguments

- x:

  Numeric matrix-like object or MLX array with observations in rows.

- retx:

  Should the rotated scores be returned?

- center, scale.:

  Passed to [`base::scale()`](https://rdrr.io/r/base/scale.html).
  User-supplied vectors are supported.

- tol:

  Optional tolerance for omitting components with small standard
  deviations, relative to the leading component.

- rank.:

  Optional maximal rank. If smaller than `min(n, p)`, the fit uses the
  randomized truncated PCA path.

- oversample:

  Oversampling added to the randomized subspace dimension. Ignored for
  exact fits.

- n_iter:

  Number of randomized power iterations. Ignored for exact fits.

- seed:

  Seed used for the randomized projection basis. Ignored for exact fits.

- ...:

  Additional arguments are rejected for compatibility with
  [`stats::prcomp()`](https://rdrr.io/r/stats/prcomp.html).

## Value

An object of class `c("mlxs_prcomp", "prcomp")`.

## Details

The interface follows
[`stats::prcomp()`](https://rdrr.io/r/stats/prcomp.html) closely.
Full-rank fits use an exact decomposition. When `rank.` is supplied and
smaller than `min(nrow(x), ncol(x))`, a randomized truncated PCA path is
used instead.
