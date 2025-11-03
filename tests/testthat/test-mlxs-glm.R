test_that("mlxs_glm gaussian matches stats::glm", {
  skip_if_not_installed("Rmlx")

  formula <- mpg ~ cyl + disp
  base_fit <- glm(formula, data = mtcars, family = gaussian())
  mlx_fit <- mlxs_glm(formula, data = mtcars, family = mlxs_gaussian())

  expect_true(mlx_fit$converged)
  expect_equal(coef(mlx_fit), coef(base_fit), tolerance = 1e-6)
  expect_equal(mlx_fit$fitted.values, fitted(base_fit), tolerance = 1e-6)
  expect_equal(mlx_fit$deviance, base_fit$deviance, tolerance = 1e-6)
  expect_equal(unname(vcov(mlx_fit)), unname(vcov(base_fit)), tolerance = 1e-6)

  newdata <- head(mtcars)
  expect_equal(
    predict(mlx_fit, newdata = newdata, type = "response"),
    predict(base_fit, newdata = newdata, type = "response"),
    tolerance = 1e-6
  )

  pearson_resid <- residuals(mlx_fit, type = "pearson")
  expect_equal(unname(pearson_resid), unname(residuals(base_fit, type = "pearson")), tolerance = 1e-5)

  tidy_df <- tidy(mlx_fit)
  expect_s3_class(tidy_df, "data.frame")
  expect_equal(unname(tidy_df$estimate), unname(coef(base_fit)[tidy_df$term]), tolerance = 1e-6)

  glance_df <- glance(mlx_fit)
  expect_equal(glance_df$aic, base_fit$aic, tolerance = 1e-6)

  aug_df <- augment(mlx_fit)
  expect_equal(unname(aug_df$.fitted), unname(fitted(base_fit)), tolerance = 1e-6)

  aug_new <- augment(mlx_fit, newdata = newdata)
  expect_equal(unname(aug_new$.fitted), unname(predict(base_fit, newdata = newdata, type = "response")), tolerance = 1e-6)

  expect_s3_class(summary(mlx_fit), "summary.mlxs_glm")
  expect_equal(anova(mlx_fit)$Deviance, anova(base_fit)$Deviance, tolerance = 1e-6)
})

test_that("mlxs_glm binomial matches stats::glm", {
  skip_if_not_installed("Rmlx")

  data <- mtcars
  data$vs <- ifelse(data$vs > 0, 1, 0)
  formula <- vs ~ mpg + wt

  base_fit <- glm(formula, data = data, family = binomial())
  mlx_fit <- mlxs_glm(formula, data = data, family = mlxs_binomial())

  expect_true(mlx_fit$converged)
  expect_equal(coef(mlx_fit), coef(base_fit), tolerance = 1e-5)
  expect_equal(unname(mlx_fit$fitted.values), as.vector(fitted(base_fit)), tolerance = 1e-5)
  expect_equal(mlx_fit$deviance, base_fit$deviance, tolerance = 1e-5)

  newdata <- head(data)
  expect_equal(
    predict(mlx_fit, newdata = newdata, type = "response"),
    predict(base_fit, newdata = newdata, type = "response"),
    tolerance = 1e-5
  )

  dev_res <- residuals(mlx_fit, type = "deviance")
  expect_equal(unname(dev_res), unname(residuals(base_fit, type = "deviance")), tolerance = 1e-5)

  tidy_df <- tidy(mlx_fit)
  expect_equal(unname(tidy_df$estimate), unname(coef(base_fit)[tidy_df$term]), tolerance = 1e-5)

  glance_df <- glance(mlx_fit)
  expect_equal(glance_df$deviance, base_fit$deviance, tolerance = 1e-5)

  aug_df <- augment(mlx_fit)
  expect_equal(unname(aug_df$.fitted), unname(fitted(base_fit)), tolerance = 1e-5)
})
