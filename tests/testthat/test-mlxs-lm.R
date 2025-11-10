test_that("mlxs_lm matches stats::lm coefficients and fitted values", {
  formula <- mpg ~ cyl + disp
  subset_expr <- mtcars$mpg > 20

  base_fit <- lm(formula, data = mtcars, subset = subset_expr)
  mlx_fit <- mlxs_lm(formula, data = mtcars, subset = subset_expr)

  expect_equal(
    drop(as.matrix(mlx_fit$coefficients)),
    base_fit$coefficients,
    tolerance = 1e-6,
    ignore_attr = TRUE
  )

  expect_equal(
    drop(as.matrix(mlx_fit$fitted.values)),
    base_fit$fitted.values,
    tolerance = 1e-6,
    ignore_attr = TRUE
  )

  expect_equal(
    drop(as.matrix(mlx_fit$residuals)),
    base_fit$residuals,
    tolerance = 1e-6,
    ignore_attr = TRUE
  )

  expect_equal(drop(as.matrix(coef(mlx_fit))), coef(base_fit), tolerance = 1e-6, ignore_attr = TRUE)
  expect_equal(drop(as.matrix(fitted(mlx_fit))), fitted(base_fit), tolerance = 1e-6, ignore_attr = TRUE)
  expect_equal(drop(as.matrix(residuals(mlx_fit))), residuals(base_fit), tolerance = 1e-6, ignore_attr = TRUE)
  expect_equal(as.matrix(vcov(mlx_fit)), vcov(base_fit), tolerance = 1e-6, ignore_attr = TRUE)
  expect_equal(confint(mlx_fit), confint(base_fit), tolerance = 1e-6, ignore_attr = TRUE)
  expect_equal(nobs(mlx_fit), nobs(base_fit))

  newdata <- head(mtcars, 5)
  expect_equal(
    drop(as.matrix(predict(mlx_fit, newdata = newdata))),
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

  augment_mlx <- augment(mlx_fit, output = "mlx")
  expect_s3_class(augment_mlx$.fitted, "mlx")
  expect_s3_class(augment_mlx$.resid, "mlx")

  expect_equal(model.frame(mlx_fit), model.frame(base_fit))
  expect_equal(model.matrix(mlx_fit), model.matrix(base_fit), tolerance = 1e-6, ignore_attr = TRUE)
  expect_equal(model.frame(mlx_fit)[[1]], model.response(model.frame(base_fit)), ignore_attr = TRUE)
  expect_equal(terms(mlx_fit), terms(base_fit))
  expect_error(anova(mlx_fit), "not implemented", fixed = TRUE)

  updated <- update(mlx_fit, . ~ . + wt)
  updated_base <- update(base_fit, . ~ . + wt)
  expect_equal(drop(as.matrix(coef(updated))), coef(updated_base), tolerance = 1e-6, ignore_attr = TRUE)

  sum_obj <- summary(mlx_fit)
  expect_s3_class(sum_obj, "summary.mlxs_lm")
  expect_equal(drop(as.matrix(sum_obj$coef)), coef(base_fit), tolerance = 1e-6, ignore_attr = TRUE)
})

test_that("mlxs_lm handles weights like stats::lm", {
  formula <- mpg ~ cyl + disp
  w <- seq_len(nrow(mtcars)) / nrow(mtcars)

  base_fit <- lm(formula, data = mtcars, weights = w)
  mlx_fit <- mlxs_lm(formula, data = mtcars, weights = w)

  expect_equal(drop(as.matrix(mlx_fit$coefficients)), coef(base_fit), tolerance = 1e-6, ignore_attr = TRUE)
  expect_equal(drop(as.matrix(mlx_fit$fitted.values)), fitted(base_fit), tolerance = 1e-6, ignore_attr = TRUE)
  expect_equal(drop(as.matrix(mlx_fit$residuals)), residuals(base_fit), tolerance = 1e-6, ignore_attr = TRUE)
})

test_that("mlxs_lm bootstrap summary provides se", {
  fit <- mlxs_lm(mpg ~ cyl + disp, data = mtcars)
  sum_boot <- summary(fit, bootstrap = TRUE, bootstrap_args = list(B = 20, seed = 123, progress = FALSE))
  expect_true(!is.null(sum_boot$bootstrap))
  expect_equal(length(sum_boot$bootstrap$se), length(drop(as.matrix(coef(fit)))))
  tidy_boot <- tidy(fit, bootstrap = TRUE, bootstrap_args = list(B = 15, seed = 123, progress = FALSE))
  expect_true(all(!is.na(tidy_boot$std.error)))

  sum_resid <- summary(fit,
                       bootstrap = TRUE,
                       bootstrap_args = list(bootstrap_type = "residual", B = 10, seed = 321, progress = FALSE))
  expect_true(!is.null(sum_resid$bootstrap))
  expect_equal(length(sum_resid$bootstrap$se), length(drop(as.matrix(coef(fit)))))
})
