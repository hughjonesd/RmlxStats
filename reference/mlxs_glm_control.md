# Control parameters

Control parameters

## Usage

``` r
mlxs_glm_control(
  epsilon = 1e-08,
  epsilon_f64 = 1e-06,
  maxit = 25,
  trace = FALSE
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

## Value

A list with default values filled in.
