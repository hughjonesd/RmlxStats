# Control parameters

Control parameters

## Usage

``` r
mlxs_glm_control(
  epsilon = 1e-08,
  epsilon_f64 = 1e-06,
  maxit = 25,
  trace = FALSE,
  rank_tol = NULL
)
```

## Arguments

- epsilon:

  Convergence tolerance parameter, interpreted as in
  [`stats::glm.control()`](https://rdrr.io/r/stats/glm.control.html).
  Iterations converge when
  `abs(deviance - deviance_old)/(abs(deviance) + 0.1) < epsilon`.

- epsilon_f64:

  Move operations to float64 on the cpu when convergence is this close
  (using the same expression as above). Doing this allows more precision
  but slows computation.

- maxit:

  Maximum number of IWLS iterations.

- trace:

  Logical: trace each iteration?

- rank_tol:

  Optional relative tolerance used to detect rank-deficient systems.
  `NULL` uses the package default, which varies by dtype and is 1e-6 for
  float32 matrices. Set to `FALSE` to skip rank checks entirely. Note
  that higher numbers indicate *lower* tolerance.

## Value

A list with default values filled in.
