# mlxs_glm method utilities

Support functions that provide a familiar S3 surface for `mlxs_glm` fits
by delegating to equivalent base `glm` behaviour where helpful.

## Usage

``` r
# S3 method for class 'mlxs_glm'
coef(object, ...)

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
print(x, digits = max(3, getOption("digits") - 3), ...)

# S3 method for class 'mlxs_glm'
summary(object, ...)

# S3 method for class 'summary.mlxs_glm'
print(x, ...)

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
  ...
)
```

## Arguments

- object:

  An `mlxs_glm` model fit.

- ...:

  Additional arguments passed to underlying methods.

- newdata:

  Optional data frame used for prediction.

- type:

  Character string indicating the scale of the prediction or residuals
  to return.

- se.fit:

  Logical. Should standard errors of the fit be returned when supported?

- x:

  An `mlxs_glm` model fit (for methods with a leading `x` argument).

- digits:

  Number of significant digits to print for summaries.

- formula, data:

  Optional formula and data overrides used by `augment.mlxs_glm()`.

- type.predict, type.residuals:

  Character strings controlling the scale of fitted values and residuals
  returned by `augment.mlxs_glm()`.

- se_fit:

  Logical; standard-error analogue for `augment`.
