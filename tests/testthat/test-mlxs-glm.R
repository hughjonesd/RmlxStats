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
  expect_equal(hatvalues(mlx_fit), hatvalues(base_fit), tolerance = 1e-6, ignore_attr = TRUE)

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
    rownames(coef(updated, output = "mlx")),
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

test_that("mlxs_glm defaults to na.exclude and pads training predictions", {
  data <- mtcars
  data$disp[c(2, 7)] <- NA_real_
  formula <- mpg ~ cyl + disp

  base_fit <- glm(
    formula,
    data = data,
    family = gaussian(),
    na.action = stats::na.exclude
  )
  mlx_fit <- mlxs_glm(formula, data = data, family = mlxs_gaussian())

  expect_equal(
    drop(as.matrix(coef(mlx_fit))),
    coef(base_fit),
    tolerance = 1e-6,
    ignore_attr = TRUE
  )

  mlx_response_raw <- predict(mlx_fit, type = "response")
  mlx_link_raw <- predict(mlx_fit, type = "link")
  expect_s3_class(mlx_response_raw, "mlx")
  expect_s3_class(mlx_link_raw, "mlx")
  mlx_response <- as.numeric(mlx_response_raw)
  mlx_link <- as.numeric(mlx_link_raw)
  base_response <- as.numeric(predict(base_fit, type = "response"))
  base_link <- as.numeric(predict(base_fit, type = "link"))
  keep <- !is.na(base_response)

  expect_equal(length(mlx_response), nrow(data))
  expect_equal(length(mlx_link), nrow(data))
  expect_equal(is.na(mlx_response), is.na(base_response))
  expect_equal(is.na(mlx_link), is.na(base_link))
  expect_equal(mlx_response[keep], base_response[keep], tolerance = 1e-6)
  expect_equal(mlx_link[keep], base_link[keep], tolerance = 1e-6)

  mlx_resid_raw <- residuals(mlx_fit, type = "pearson")
  expect_s3_class(mlx_resid_raw, "mlx")
  mlx_resid <- as.numeric(mlx_resid_raw)
  expect_equal(length(mlx_resid), nrow(data))
  expect_equal(
    is.na(mlx_resid),
    unname(is.na(residuals(base_fit, type = "pearson")))
  )

  omit_fit <- mlxs_glm(
    formula,
    data = data,
    family = mlxs_gaussian(),
    na.action = stats::na.omit
  )
  expect_equal(length(as.numeric(predict(omit_fit))), nrow(data) - 2L)
  expect_false(anyNA(as.numeric(predict(omit_fit))))
})

test_that("mlxs_glm pads binomial predictions with na.exclude", {
  data <- transform(mtcars, vs = as.integer(vs > 0))
  data$mpg[c(3, 8)] <- NA_real_
  formula <- vs ~ mpg + wt

  base_fit <- glm(
    formula,
    data = data,
    family = binomial(),
    na.action = stats::na.exclude
  )
  mlx_fit <- mlxs_glm(formula, data = data, family = mlxs_binomial())

  mlx_response_raw <- predict(mlx_fit, type = "response")
  expect_s3_class(mlx_response_raw, "mlx")
  mlx_response <- as.numeric(mlx_response_raw)
  base_response <- as.numeric(predict(base_fit, type = "response"))
  keep <- !is.na(base_response)
  expect_equal(length(mlx_response), nrow(data))
  expect_equal(is.na(mlx_response), is.na(base_response))
  expect_equal(mlx_response[keep], base_response[keep], tolerance = 1e-5)
})

test_that("mlxs_glm augment uses complete-case internals with na.exclude", {
  data <- mtcars
  data$mpg[1] <- NA_real_

  fit <- mlxs_glm(gear ~ mpg, data = data, family = mlxs_gaussian())
  base_fit <- glm(
    gear ~ mpg,
    data = data,
    family = gaussian(),
    na.action = stats::na.exclude
  )

  expect_output(print(fit), "Residual deviance")
  expect_s3_class(summary(fit), "summary.mlxs_glm")
  aug <- augment(fit)
  expect_equal(nrow(aug), nrow(model.frame(fit)))
  expect_equal(aug$.fitted, unname(fitted(base_fit)[-1]), tolerance = 1e-6)
  expect_equal(
    aug$.resid,
    unname(residuals(base_fit, type = "response")[-1]),
    tolerance = 1e-6
  )
  expect_equal(glance(fit)$deviance, base_fit$deviance, tolerance = 1e-6)
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
  sum_boot <- summary(
    fit,
    bootstrap = TRUE,
    bootstrap_args = list(B = 15, seed = 42, progress = FALSE)
  )
  expect_true(!is.null(sum_boot$bootstrap))
  expect_equal(length(sum_boot$bootstrap$se), length(coef(fit)))
  expect_null(sum_boot$confint)

  sum_boot_ci <- summary(
    fit,
    bootstrap = TRUE,
    confint = TRUE,
    level = 0.9,
    bootstrap_args = list(B = 15, seed = 42, progress = FALSE)
  )
  expect_equal(dim(sum_boot_ci$confint), c(3L, 2L))
  expect_equal(rownames(sum_boot_ci$confint), 
               rownames(coef(fit, output = "mlx")))
  expect_equal(colnames(sum_boot_ci$confint), c("5 %", "95 %"))
  expect_output(print(sum_boot_ci), "5 %")
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

  sum_ci <- summary(fit, confint = TRUE, level = 0.9)
  expect_equal(
    sum_ci$confint,
    confint(fit, level = 0.9),
    tolerance = 1e-6,
    ignore_attr = TRUE
  )
})

test_that("confint.mlxs_glm supports bootstrap percentile intervals", {
  data <- transform(mtcars, vs = as.integer(vs > 0))
  fit <- mlxs_glm(vs ~ mpg + wt, data = data, family = mlxs_binomial())
  args <- list(B = 12, seed = 42, progress = FALSE)

  ci <- confint(fit, bootstrap = TRUE, bootstrap_args = args)
  expect_equal(dim(ci), c(3L, 2L))
  expect_equal(rownames(ci), rownames(coef(fit, output = "mlx")))
  expect_equal(colnames(ci), c("2.5 %", "97.5 %"))
  expect_true(all(is.finite(ci)))
  expect_true(all(ci[, 1L] <= ci[, 2L]))

  ci_again <- confint(fit, bootstrap = TRUE, bootstrap_args = args)
  expect_equal(ci_again, ci)

  ci_wt <- confint(
    fit,
    parm = "wt",
    bootstrap = TRUE,
    bootstrap_args = args
  )
  expect_equal(dim(ci_wt), c(1L, 2L))
  expect_equal(rownames(ci_wt), "wt")

  expect_error(
    confint(
      fit,
      bootstrap = TRUE,
      bootstrap_args = list(
        bootstrap_type = "residual",
        B = 10,
        seed = 11,
        progress = FALSE
      )
    ),
    "supports only gaussian/quasigaussian",
    fixed = TRUE
  )
})

test_that("confint.mlxs_glm supports gaussian residual bootstrap", {
  fit <- mlxs_glm(mpg ~ cyl + disp + wt, data = mtcars, family = mlxs_gaussian())
  ci <- confint(
    fit,
    parm = "wt",
    bootstrap = TRUE,
    bootstrap_args = list(
      bootstrap_type = "residual",
      B = 10,
      seed = 11,
      progress = FALSE
    )
  )
  expect_equal(dim(ci), c(1L, 2L))
  expect_equal(rownames(ci), "wt")
  expect_true(all(is.finite(ci)))
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
  rownames(expected_ci) <- rownames(coef(mlx_fit, output = "mlx"))
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
