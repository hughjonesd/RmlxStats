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
  oversample = NULL,
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

  Oversampling added to the randomized subspace dimension. If `NULL`,
  randomized fits use `min(rank., max(10, ceiling(rank. / 2)))`. Ignored
  for exact fits.

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

The randomized path first sketches a slightly larger subspace than the
requested rank, then compresses back down to the requested components.
The `oversample` parameter controls how much extra space is used in that
sketch: larger values make it less likely that the random sketch misses
part of the leading principal subspace. The `n_iter` parameter applies
additional power iterations, which improve accuracy when the singular
values decay slowly but require extra passes over the matrix.

By default, `oversample` is chosen as
`min(rank., max(10, ceiling(rank. / 2)))`, which keeps the usual
constant-size oversampling for small target ranks while allowing more
slack for larger truncated fits. This follows common randomized SVD
guidance to start with modest oversampling, often around 5 to 10, and to
increase oversampling before increasing the number of power iterations.

## References

Halko, N., Martinsson, P.-G., and Tropp, J. A. (2011). Finding Structure
with Randomness: Probabilistic Algorithms for Constructing Approximate
Matrix Decompositions. *SIAM Review*, 53(2), 217-288.

Musco, C. and Musco, C. (2015). Randomized Block Krylov Methods for
Stronger and Faster Approximate Singular Value Decomposition. *NeurIPS
2015*.
