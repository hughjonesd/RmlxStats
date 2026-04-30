test_that("mlxs_glm gaussian matches stats::glm", {
  formula <- mpg ~ cyl + disp
  base_fit <- glm(formula, data = mtcars, family = gaussian())
  mlx_fit <- mlxs_glm(formula, data = mtcars, family = mlxs_gaussian())

  expect_true(mlx_fit$converged)
  expect_equal(drop(as.matrix(coef(mlx_fit))), coef(base_fit), tolerance = 1e-6, ignore_attr = TRUE)
  expect_equal(
    drop(as.matrix(mlx_fit$fitted.values)),
    fitted(base_fit),
    tolerance = 1e-6,
    ignore_attr = TRUE
  )
  expect_equal(mlx_fit$deviance, base_fit$deviance, tolerance = 1e-6)
  expect_equal(unname(as.matrix(vcov(mlx_fit))), unname(vcov(base_fit)), tolerance = 1e-6)
  expect_equal(
    confint(mlx_fit),
    confint.default(base_fit),
    tolerance = 1e-6,
    ignore_attr = TRUE
  )

  newdata <- head(mtcars)
  expect_equal(
    drop(as.matrix(predict(mlx_fit, newdata = newdata, type = "response"))),
    predict(base_fit, newdata = newdata, type = "response"),
    tolerance = 1e-6,
    ignore_attr = TRUE
  )

  pearson_resid <- drop(as.matrix(residuals(mlx_fit, type = "pearson")))
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
  augment_mlx <- augment(mlx_fit, output = "mlx")
  expect_s3_class(augment_mlx$.fitted, "mlx")
  expect_s3_class(augment_mlx$.resid, "mlx")

  expect_s3_class(summary(mlx_fit), "summary.mlxs_glm")
  expect_error(anova(mlx_fit), "not implemented", fixed = TRUE)
})

test_that("mlxs_glm updates through the mlxs_model superclass", {
  fit <- mlxs_glm(mpg ~ cyl + disp, data = mtcars, family = mlxs_gaussian())
  updated <- update(fit, . ~ . + hp)

  expect_s3_class(updated, "mlxs_glm")
  expect_equal(
    .mlxs_coef_names(updated),
    c("(Intercept)", "cyl", "disp", "hp")
  )
})

test_that("mlxs_glm respects observation weights", {
  formula <- mpg ~ cyl + disp
  w <- seq_len(nrow(mtcars)) / nrow(mtcars)

  base_fit <- glm(formula, data = mtcars, family = gaussian(), weights = w)
  mlx_fit <- mlxs_glm(formula, data = mtcars, family = mlxs_gaussian(), weights = w)

  expect_true(mlx_fit$converged)
  expect_equal(drop(as.matrix(coef(mlx_fit))), coef(base_fit), tolerance = 1e-5, ignore_attr = TRUE)
  expect_equal(
    drop(as.matrix(mlx_fit$fitted.values)),
    fitted(base_fit),
    tolerance = 1e-5,
    ignore_attr = TRUE
  )
  expect_equal(mlx_fit$deviance, base_fit$deviance, tolerance = 1e-5)
  expect_equal(
    drop(as.matrix(mlx_fit$weights)),
    unname(base_fit$prior.weights),
    tolerance = 1e-12
  )
})

test_that("mlxs_glm rejects rank-deficient model matrices", {
  gaussian_data <- mtcars
  gaussian_data$disp_copy <- gaussian_data$disp
  gaussian_data$linear_combo <- gaussian_data$cyl + gaussian_data$disp

  expect_error(
    mlxs_glm(
      mpg ~ cyl + disp + disp_copy,
      data = gaussian_data,
      family = mlxs_gaussian()
    ),
    "full-rank model matrix",
    fixed = TRUE
  )
  expect_error(
    mlxs_glm(
      mpg ~ cyl + disp + linear_combo,
      data = gaussian_data,
      family = mlxs_gaussian()
    ),
    "full-rank model matrix",
    fixed = TRUE
  )

  binomial_data <- transform(mtcars, vs = as.integer(vs > 0))
  binomial_data$mpg_copy <- binomial_data$mpg
  expect_error(
    mlxs_glm(
      vs ~ mpg + wt + mpg_copy,
      data = binomial_data,
      family = mlxs_binomial()
    ),
    "full-rank model matrix",
    fixed = TRUE
  )
})

test_that("mlxs_glm bootstrap summary works", {
  formula <- vs ~ mpg + wt
  data <- transform(mtcars, vs = as.integer(vs > 0))
  fit <- mlxs_glm(formula, data = data, family = mlxs_binomial())
  sum_boot <- summary(fit, bootstrap = TRUE, bootstrap_args = list(B = 15, seed = 42, progress = FALSE))
  expect_true(!is.null(sum_boot$bootstrap))
  expect_equal(length(sum_boot$bootstrap$se), length(coef(fit)))
  tidy_boot <- tidy(fit, bootstrap = TRUE, bootstrap_args = list(B = 12, seed = 42, progress = FALSE))
  expect_true(all(!is.na(tidy_boot$std.error)))
})

test_that("mlxs_glm residual bootstrap works for gaussian", {
  formula <- mpg ~ cyl + disp + wt
  fit <- mlxs_glm(formula, data = mtcars, family = mlxs_gaussian())
  sum_resid <- summary(
    fit,
    bootstrap = TRUE,
    bootstrap_args = list(bootstrap_type = "residual", B = 10, seed = 11, progress = FALSE)
  )
  expect_true(!is.null(sum_resid$bootstrap))
  expect_equal(length(sum_resid$bootstrap$se), length(coef(fit)))
})

test_that("mlxs_glm binomial matches stats::glm", {
  data <- mtcars
  data$vs <- ifelse(data$vs > 0, 1, 0)
  formula <- vs ~ mpg + wt

  base_fit <- glm(formula, data = data, family = binomial())
  mlx_fit <- mlxs_glm(formula, data = data, family = mlxs_binomial())

  expect_true(mlx_fit$converged)
  expect_equal(drop(as.matrix(coef(mlx_fit))), coef(base_fit), tolerance = 1e-5, ignore_attr = TRUE)
  expect_equal(unname(drop(as.matrix(mlx_fit$fitted.values))), as.vector(fitted(base_fit)), tolerance = 1e-5)
  expect_equal(mlx_fit$deviance, base_fit$deviance, tolerance = 1e-5)
  se <- as.numeric(sqrt(Rmlx::diag(vcov(mlx_fit))))
  est <- as.numeric(coef(mlx_fit))
  expected_ci <- cbind(
    est + se * qnorm(0.025),
    est + se * qnorm(0.975)
  )
  rownames(expected_ci) <- .mlxs_coef_names(mlx_fit)
  colnames(expected_ci) <- c("2.5 %", "97.5 %")
  expect_equal(confint(mlx_fit), expected_ci, tolerance = 1e-12)

  newdata <- head(data)
  expect_equal(
    drop(as.matrix(predict(mlx_fit, newdata = newdata, type = "response"))),
    predict(base_fit, newdata = newdata, type = "response"),
    tolerance = 1e-5,
    ignore_attr = TRUE
  )

  dev_res <- drop(as.matrix(residuals(mlx_fit, type = "deviance")))
  expect_equal(unname(dev_res), unname(residuals(base_fit, type = "deviance")), tolerance = 1e-5)

  tidy_df <- tidy(mlx_fit)
  expect_equal(unname(tidy_df$estimate), unname(coef(base_fit)[tidy_df$term]), tolerance = 1e-5)

  glance_df <- glance(mlx_fit)
  expect_equal(glance_df$deviance, base_fit$deviance, tolerance = 1e-5)

  aug_df <- augment(mlx_fit)
  expect_equal(unname(aug_df$.fitted), unname(fitted(base_fit)), tolerance = 1e-5)
})

test_that("mlxs_glm poisson matches stats::glm", {
  data <- mtcars
  data$cyl_count <- round(abs(data$cyl + rnorm(nrow(data), sd = 0.25)))
  formula <- cyl_count ~ mpg + wt

  base_fit <- glm(formula, data = data, family = poisson())
  mlx_fit <- mlxs_glm(formula, data = data, family = mlxs_poisson())

  expect_true(mlx_fit$converged)
  expect_equal(drop(as.matrix(coef(mlx_fit))), coef(base_fit), tolerance = 1e-5, ignore_attr = TRUE)
  expect_equal(unname(drop(as.matrix(mlx_fit$fitted.values))), as.vector(fitted(base_fit)), tolerance = 1e-5)
  expect_equal(mlx_fit$deviance, base_fit$deviance, tolerance = 1e-5)

  newdata <- head(data)
  expect_equal(
    drop(as.matrix(predict(mlx_fit, newdata = newdata, type = "response"))),
    predict(base_fit, newdata = newdata, type = "response"),
    tolerance = 1e-5,
    ignore_attr = TRUE
  )
})
