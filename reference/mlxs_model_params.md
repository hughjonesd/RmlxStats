# Shared model parameter documentation

Shared model parameter documentation

## Arguments

- formula:

  Model formula.

- data:

  Optional data frame, tibble, or environment containing the variables
  in the model.

- subset:

  Optional expression for subsetting observations.

- weights:

  Optional non-negative observation weights.

- na.action:

  How to handle missing values.

- rank_tol:

  Optional relative tolerance used to detect rank-deficient systems.
  `NULL` uses the package default, which varies by dtype and is 1e-6 for
  float32 matrices. Set to `FALSE` to skip rank checks entirely. Note
  that higher numbers indicate *lower* tolerance.
