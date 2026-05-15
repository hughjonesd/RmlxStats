# mlxs_glm method utilities

Support functions that provide a familiar S3 surface for `mlxs_glm` fits
by delegating to equivalent base `glm` behaviour where helpful.

## Usage

``` r
# S3 method for class 'mlxs_glm'
coef(object, ...)

# S3 method for class 'mlxs_glm'
weights(object, type = c("prior", "working"), ...)

# S3 method for class 'mlxs_glm'
predict(
  object,
  newdata = NULL,
  type = c("link", "response"),
  se.fit = FALSE,
  ...
)

# S3 method for class 'mlxs_glm'
fitted(object, ...)

# S3 method for class 'mlxs_glm'
residuals(object, type = c("deviance", "pearson", "working", "response"), ...)

# S3 method for class 'mlxs_glm'
vcov(object, ...)

# S3 method for class 'mlxs_glm'
confint(
  object,
  parm,
  level = 0.95,
  ...,
  bootstrap = FALSE,
  bootstrap_args = list(B = 200L, seed = NULL, progress = FALSE, bootstrap_type = "case")
)

# S3 method for class 'mlxs_glm'
print(x, digits = max(3, getOption("digits") - 3), ...)

# S3 method for class 'mlxs_glm'
summary(
  object,
  bootstrap = FALSE,
  bootstrap_args = list(B = 200L, seed = NULL, progress = FALSE, bootstrap_type = "case"),
  confint = FALSE,
  level = 0.95,
  ...
)

# S3 method for class 'summary.mlxs_glm'
print(x, digits = max(3, getOption("digits") - 3), ...)

# S3 method for class 'mlxs_glm'
anova(object, ...)

# S3 method for class 'mlxs_glm'
model.frame(formula, ...)

# S3 method for class 'mlxs_glm'
model.matrix(object, ...)

# S3 method for class 'mlxs_glm'
terms(x, ...)

# S3 method for class 'mlxs_glm'
nobs(object, ...)

# S3 method for class 'mlxs_glm'
tidy(x, ...)

# S3 method for class 'mlxs_glm'
glance(x, ...)

# S3 method for class 'mlxs_glm'
augment(
  x,
  data = x$model,
  newdata = NULL,
  type.predict = c("response", "link"),
  type.residuals = c("response", "deviance"),
  se_fit = FALSE,
  output = c("data.frame", "mlx"),
  ...
)

# S3 method for class 'mlxs_glm'
estfun(x, ..., output = c("matrix", "mlx"))

# S3 method for class 'mlxs_glm'
hatvalues(model, ..., output = c("matrix", "mlx"))

# S3 method for class 'mlxs_glm'
bread(x, ...)
```

## Arguments

- object, model:

  An `mlxs_glm` model fit.

- ...:

  Additional arguments passed to underlying methods.

- type:

  Character string indicating the scale of the prediction or residuals
  to return.

- newdata:

  Optional data frame used for prediction.

- se.fit:

  Logical. Should standard errors of the fit be returned when supported?

- parm:

  Parameter specification for confidence intervals.

- level:

  Confidence level for intervals.

- bootstrap:

  Logical; should bootstrap standard errors or confidence intervals be
  computed?

- bootstrap_args:

  List of bootstrap configuration options. See
  [`mlxs_boot()`](https://hughjonesd.github.io/RmlxStats/reference/mlxs_boot.md).

- x:

  An `mlxs_glm` model fit (for methods with a leading `x` argument).

- digits:

  Number of significant digits to print for summaries.

- confint:

  Logical; should confidence intervals be included in the summary
  object?

- formula, data:

  Optional formula and data overrides used by `augment.mlxs_glm()`.

- type.predict, type.residuals:

  Character strings controlling the scale of fitted values and residuals
  returned by `augment.mlxs_glm()`.

- se_fit:

  Logical; standard-error analogue for `augment`.

- output:

  Character string; return format ("data.frame" or "mlx").
