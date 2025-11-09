test_that("mlxs_lm matches stats::lm coefficients and fitted values", {

  tryCatch(
    Rmlx::as_mlx(matrix(1, nrow = 1)),
    error = function(e) {
      skip(paste("Rmlx tensors unavailable:", e$message))
    }
  )

  formula <- mpg ~ cyl + disp
  subset_expr <- mtcars$mpg > 20

  base_fit <- lm(formula, data = mtcars, subset = subset_expr)
  mlx_fit <- mlxs_lm(formula, data = mtcars, subset = subset_expr)

  as_vec <- function(x) {
    if (inherits(x, "mlx")) {
      drop(as.matrix(x))
    } else {
      x
    }
  }

  expect_equal(
    as_vec(mlx_fit$coefficients),
    base_fit$coefficients,
    tolerance = 1e-6,
    ignore_attr = TRUE
  )

  expect_equal(
    as_vec(mlx_fit$fitted.values),
    base_fit$fitted.values,
    tolerance = 1e-6,
    ignore_attr = TRUE
  )

  expect_equal(
    as_vec(mlx_fit$residuals),
    base_fit$residuals,
    tolerance = 1e-6,
    ignore_attr = TRUE
  )

  expect_equal(coef(mlx_fit), coef(base_fit), tolerance = 1e-6)
  expect_equal(as_vec(fitted(mlx_fit)), fitted(base_fit), tolerance = 1e-6, ignore_attr = TRUE)
  expect_equal(as_vec(residuals(mlx_fit)), residuals(base_fit), tolerance = 1e-6, ignore_attr = TRUE)
  expect_equal(vcov(mlx_fit), vcov(base_fit), tolerance = 1e-6, ignore_attr = TRUE)
  expect_equal(confint(mlx_fit), confint(base_fit), tolerance = 1e-6, ignore_attr = TRUE)
  expect_equal(nobs(mlx_fit), nobs(base_fit))

  newdata <- head(mtcars, 5)
  expect_equal(
    as_vec(predict(mlx_fit, newdata = newdata)),
    predict(base_fit, newdata = newdata),
    tolerance = 1e-6,
    ignore_attr = TRUE
  )

  tidy_df <- tidy(mlx_fit)
  expect_equal(
    tidy_df$estimate,
    unname(coef(base_fit)[tidy_df$term]),
    tolerance = 1e-6
  )
  expect_equal(
    tidy_df$std.error,
    unname(sqrt(diag(vcov(base_fit)))[tidy_df$term]),
    tolerance = 1e-6
  )

  glance_df <- glance(mlx_fit)
  base_summary <- summary(base_fit)
  expect_equal(glance_df$r.squared, base_summary$r.squared, tolerance = 1e-6)
  expect_equal(glance_df$adj.r.squared, base_summary$adj.r.squared, tolerance = 1e-6)
  expect_equal(glance_df$sigma, base_summary$sigma, tolerance = 1e-6)

  augment_df <- augment(mlx_fit)
  expect_equal(augment_df$.fitted, fitted(base_fit), tolerance = 1e-6, ignore_attr = TRUE)
  expect_equal(augment_df$.resid, residuals(base_fit), tolerance = 1e-6, ignore_attr = TRUE)

  augment_new <- augment(mlx_fit, newdata = newdata)
  expect_equal(
    augment_new$.fitted,
    predict(base_fit, newdata = newdata),
    tolerance = 1e-6,
    ignore_attr = TRUE
  )
  expect_false(".resid" %in% names(augment_new))

  expect_equal(model.frame(mlx_fit), model.frame(base_fit))
  expect_equal(model.matrix(mlx_fit), model.matrix(base_fit), tolerance = 1e-6, ignore_attr = TRUE)
  expect_equal(model.frame(mlx_fit)[[1]], model.response(model.frame(base_fit)), ignore_attr = TRUE)
  expect_equal(terms(mlx_fit), terms(base_fit))
  expect_equal(anova(mlx_fit), anova(base_fit), tolerance = 1e-6, ignore_attr = TRUE)

  updated <- update(mlx_fit, . ~ . + wt)
  updated_base <- update(base_fit, . ~ . + wt)
  expect_equal(coef(updated), coef(updated_base), tolerance = 1e-6)

  sum_obj <- summary(mlx_fit)
  expect_s3_class(sum_obj, "summary.mlxs_lm")
  expect_equal(sum_obj$coefficients[, "Estimate"], coef(base_fit), tolerance = 1e-6)
})
