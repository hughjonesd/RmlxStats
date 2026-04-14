# mlxs_lm method utilities

These helpers provide the familiar S3 surface for `mlxs_lm` fits.

## Arguments

- object:

  An `mlxs_lm` model fit.

- x:

  An `mlxs_lm` model fit (for methods with a leading `x` argument).

- ...:

  Additional arguments passed to underlying methods.

- newdata:

  Optional data frame for prediction.

- parm:

  Parameter specification for confidence intervals.

- level:

  Confidence level for intervals.

- bootstrap:

  Logical; should bootstrap standard errors be computed?

- bootstrap_args:

  List of bootstrap configuration options.

- evaluate:

  Logical; evaluate the updated call?

- formula:

  An `mlxs_lm` object used in place of formula for `model.frame`.

- data:

  Optional data frame for `augment`.

- se_fit:

  Logical; should standard errors of fit be included?

- output:

  Character string; return format ("data.frame" or "mlx").

- row.names:

  Optional row names for data frame conversion.

- optional:

  Logical; passed to `as.data.frame`.

- digits:

  Number of significant digits for printing.
