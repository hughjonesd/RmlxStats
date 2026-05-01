# mlxs_lm method utilities

These helpers provide the familiar S3 surface for `mlxs_lm` fits.

## Usage

``` r
# S3 method for class 'mlxs_lm'
coef(object, ...)

# S3 method for class 'mlxs_lm'
predict(object, newdata = NULL, ...)

# S3 method for class 'mlxs_lm'
fitted(object, ...)

# S3 method for class 'mlxs_lm'
residuals(object, ...)

# S3 method for class 'mlxs_lm'
vcov(object, ...)

# S3 method for class 'mlxs_lm'
confint(object, parm, level = 0.95, ...)

# S3 method for class 'mlxs_lm'
anova(object, ...)

# S3 method for class 'mlxs_anova'
as.data.frame(x, row.names = NULL, optional = FALSE, ...)

# S3 method for class 'mlxs_anova'
print(x, ...)

# S3 method for class 'mlxs_anova'
tidy(x, ...)

# S3 method for class 'mlxs_lm'
summary(object, bootstrap = FALSE, bootstrap_args = list(), ...)

# S3 method for class 'mlxs_lm'
print(x, ...)

# S3 method for class 'summary.mlxs_lm'
print(x, ...)

# S3 method for class 'mlxs_lm'
model.frame(formula, ...)

# S3 method for class 'mlxs_lm'
model.matrix(object, ...)

# S3 method for class 'mlxs_lm'
terms(x, ...)

# S3 method for class 'mlxs_lm'
nobs(object, ...)

# S3 method for class 'mlxs_lm'
tidy(x, ...)

# S3 method for class 'mlxs_lm'
glance(x, ...)

# S3 method for class 'mlxs_lm'
augment(
  x,
  data = model.frame(x),
  newdata = NULL,
  se_fit = FALSE,
  output = c("data.frame", "mlx"),
  ...
)
```

## Arguments

- object:

  An `mlxs_lm` model fit.

- ...:

  Additional arguments passed to underlying methods.

- newdata:

  Optional data frame for prediction.

- parm:

  Parameter specification for confidence intervals.

- level:

  Confidence level for intervals.

- x:

  An `mlxs_lm` model fit (for methods with a leading `x` argument).

- row.names:

  Optional row names for data frame conversion.

- optional:

  Logical; passed to `as.data.frame`.

- bootstrap:

  Logical; should bootstrap standard errors be computed?

- bootstrap_args:

  List of bootstrap configuration options. See
  [`mlxs_boot()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_boot.md).

- formula:

  An `mlxs_lm` object used in place of formula for `model.frame`.

- data:

  Optional data frame for `augment`.

- se_fit:

  Logical; should standard errors of fit be included?

- output:

  Character string; return format ("data.frame" or "mlx").
