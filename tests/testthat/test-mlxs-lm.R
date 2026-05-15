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
  expect_equal(names(coef(mlx_fit)), names(coef(base_fit)))
  expect_equal(rownames(coef(mlx_fit, output = "mlx")), names(coef(base_fit)))
  expect_equal(drop(as.matrix(fitted(mlx_fit))), fitted(base_fit), tolerance = 1e-6, ignore_attr = TRUE)
  expect_equal(drop(as.matrix(residuals(mlx_fit))), residuals(base_fit), tolerance = 1e-6, ignore_attr = TRUE)
  expect_equal(
    as.matrix(vcov(mlx_fit)),
    vcov(base_fit),
    tolerance = 1e-6,
    ignore_attr = TRUE
  )
  expect_equal(dimnames(vcov(mlx_fit)), dimnames(vcov(base_fit)))
  expect_equal(
    confint(mlx_fit),
    confint(base_fit),
    tolerance = 1e-6,
    ignore_attr = TRUE
  )
  expect_equal(
    hatvalues(mlx_fit),
    hatvalues(base_fit),
    tolerance = 1e-6, 
    ignore_attr = TRUE
  )
  expect_equal(nobs(mlx_fit), nobs(base_fit))

  updated <- update(mlx_fit, . ~ . + hp)
  expect_s3_class(updated, "mlxs_lm")
  expect_equal(
    .mlxs_coef_names(updated),
    c("(Intercept)", "cyl", "disp", "hp")
  )

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
  anova_mlx <- anova(mlx_fit)
  expect_s3_class(anova_mlx, "mlxs_anova")
  anova_df <- as.data.frame(anova_mlx)
  base_anova <- anova(base_fit)
  expect_s3_class(anova_df, "anova")
  expect_equal(rownames(anova_df), rownames(base_anova))
  expect_equal(anova_df$Df, base_anova$Df)
  expect_equal(anova_df[["Sum Sq"]], base_anova[["Sum Sq"]], tolerance = 1e-6)
  expect_equal(anova_df[["Mean Sq"]], base_anova[["Mean Sq"]], tolerance = 1e-6)
  expect_equal(anova_df[["F value"]], base_anova[["F value"]], tolerance = 1e-6, ignore_attr = TRUE)
  expect_equal(anova_df[["Pr(>F)"]], base_anova[["Pr(>F)"]], tolerance = 1e-6, ignore_attr = TRUE)
  expect_output(print(anova_mlx), "Analysis of Variance Table")
  tidy_anova <- tidy(anova_mlx)
  expect_equal(tidy_anova$term, rownames(base_anova))
  expect_equal(tidy_anova$df, base_anova$Df)
  expect_equal(tidy_anova$sumsq, base_anova[["Sum Sq"]], tolerance = 1e-6)
  expect_equal(tidy_anova$meansq, base_anova[["Mean Sq"]], tolerance = 1e-6)
  expect_equal(tidy_anova$statistic, base_anova[["F value"]], tolerance = 1e-6, ignore_attr = TRUE)
  expect_equal(tidy_anova$p.value, base_anova[["Pr(>F)"]], tolerance = 1e-6, ignore_attr = TRUE)

  updated <- update(mlx_fit, . ~ . + wt)
  updated_base <- update(base_fit, . ~ . + wt)
  expect_equal(drop(as.matrix(coef(updated))), coef(updated_base), tolerance = 1e-6, ignore_attr = TRUE)

  sum_obj <- summary(mlx_fit)
  expect_s3_class(sum_obj, "summary.mlxs_lm")
  expect_equal(drop(as.matrix(sum_obj$coef)), coef(base_fit), tolerance = 1e-6, ignore_attr = TRUE)
  expect_equal(rownames(sum_obj$coef), names(coef(base_fit)))
  expect_equal(rownames(sum_obj$std.error), names(coef(base_fit)))
  expect_equal(rownames(sum_obj$statistic), names(coef(base_fit)))
  expect_equal(rownames(sum_obj$p.value), names(coef(base_fit)))
})

test_that("mlxs_lm handles weights like stats::lm", {
  formula <- mpg ~ cyl + disp
  w <- seq_len(nrow(mtcars)) / nrow(mtcars)

  base_fit <- lm(formula, data = mtcars, weights = w)
  mlx_fit <- mlxs_lm(formula, data = mtcars, weights = w)

  expect_equal(drop(as.matrix(mlx_fit$coefficients)), coef(base_fit), tolerance = 1e-6, ignore_attr = TRUE)
  expect_equal(drop(as.matrix(mlx_fit$fitted.values)), fitted(base_fit), tolerance = 1e-6, ignore_attr = TRUE)
  expect_equal(drop(as.matrix(mlx_fit$residuals)), residuals(base_fit), tolerance = 1e-6, ignore_attr = TRUE)
  expect_equal(as.matrix(vcov(mlx_fit)), vcov(base_fit), tolerance = 1e-6, ignore_attr = TRUE)
  expect_equal(confint(mlx_fit), confint(base_fit), tolerance = 1e-6, ignore_attr = TRUE)
  expect_equal(hatvalues(mlx_fit), hatvalues(base_fit), tolerance = 1e-6, ignore_attr = TRUE)
  anova_weighted <- as.data.frame(anova(mlx_fit))
  base_weighted <- anova(base_fit)
  expect_equal(anova_weighted[["Sum Sq"]], base_weighted[["Sum Sq"]], tolerance = 1e-6)
  expect_equal(anova_weighted[["Mean Sq"]], base_weighted[["Mean Sq"]], tolerance = 1e-6)
})

test_that("mlxs_lm defaults to na.exclude and pads training predictions", {
  data <- mtcars
  data$disp[c(2, 7)] <- NA_real_
  formula <- mpg ~ cyl + disp

  base_fit <- lm(formula, data = data, na.action = stats::na.exclude)
  mlx_fit <- mlxs_lm(formula, data = data)

  expect_equal(
    drop(as.matrix(mlx_fit$coefficients)),
    coef(base_fit),
    tolerance = 1e-6,
    ignore_attr = TRUE
  )

  mlx_pred_raw <- predict(mlx_fit)
  expect_s3_class(mlx_pred_raw, "mlx")
  mlx_pred <- as.numeric(mlx_pred_raw)
  base_pred <- as.numeric(predict(base_fit))
  keep <- !is.na(base_pred)
  expect_equal(length(mlx_pred), nrow(data))
  expect_equal(is.na(mlx_pred), is.na(base_pred))
  expect_equal(mlx_pred[keep], base_pred[keep], tolerance = 1e-6)

  mlx_fitted_raw <- fitted(mlx_fit)
  mlx_resid_raw <- residuals(mlx_fit)
  expect_s3_class(mlx_fitted_raw, "mlx")
  expect_s3_class(mlx_resid_raw, "mlx")
  mlx_fitted <- as.numeric(mlx_fitted_raw)
  mlx_resid <- as.numeric(mlx_resid_raw)
  expect_equal(length(mlx_fitted), nrow(data))
  expect_equal(length(mlx_resid), nrow(data))
  expect_equal(is.na(mlx_fitted), unname(is.na(fitted(base_fit))))
  expect_equal(is.na(mlx_resid), unname(is.na(residuals(base_fit))))
  expect_equal(mlx_fitted[keep], as.numeric(fitted(base_fit))[keep],
               tolerance = 1e-6)
  expect_equal(mlx_resid[keep], as.numeric(residuals(base_fit))[keep],
               tolerance = 1e-5)

  omit_fit <- mlxs_lm(
    formula,
    data = data,
    na.action = stats::na.omit
  )
  expect_equal(length(as.numeric(predict(omit_fit))), nrow(data) - 2L)
  expect_false(anyNA(as.numeric(predict(omit_fit))))
})

test_that("mlxs_lm summaries use complete-case internals with na.exclude", {
  data <- mtcars
  data$mpg[1] <- NA_real_

  fit <- mlxs_lm(gear ~ mpg, data = data)
  base_fit <- lm(gear ~ mpg, data = data, na.action = stats::na.exclude)

  expect_output(print(fit), "Residuals")
  sum_obj <- summary(fit)
  base_summary <- summary(base_fit)
  expect_equal(sum_obj$r.squared, base_summary$r.squared, tolerance = 1e-6)
  expect_equal(
    sum_obj$adj.r.squared,
    base_summary$adj.r.squared,
    tolerance = 1e-6
  )
  expect_equal(glance(fit)$r.squared, base_summary$r.squared, tolerance = 1e-6)
})

test_that("mlxs_lm rejects rank-deficient model matrices", {
  data <- mtcars
  data$disp_copy <- data$disp
  expect_error(
    mlxs_lm(mpg ~ cyl + disp + disp_copy, data = data),
    "full-rank model matrix",
    fixed = TRUE
  )

  data$linear_combo <- data$cyl + data$disp
  expect_error(
    mlxs_lm(mpg ~ cyl + disp + linear_combo, data = data),
    "full-rank model matrix",
    fixed = TRUE
  )

  expect_error(
    mlxs_lm(
      mpg ~ cyl + disp + disp_copy,
      data = data,
      weights = seq_len(nrow(data))
    ),
    "full-rank model matrix",
    fixed = TRUE
  )
})

test_that("mlxs_lm bootstrap summary provides se", {
  fit <- mlxs_lm(mpg ~ cyl + disp, data = mtcars)
  sum_boot <- summary(
    fit,
    bootstrap = TRUE,
    bootstrap_args = list(B = 20, seed = 123, progress = FALSE)
  )
  expect_true(!is.null(sum_boot$bootstrap))
  expect_equal(length(sum_boot$bootstrap$se), length(drop(as.matrix(coef(fit)))))
  expect_null(sum_boot$confint)

  sum_boot_ci <- summary(
    fit,
    bootstrap = TRUE,
    confint = TRUE,
    level = 0.9,
    bootstrap_args = list(B = 20, seed = 123, progress = FALSE)
  )
  expect_equal(dim(sum_boot_ci$confint), c(3L, 2L))
  expect_equal(rownames(sum_boot_ci$confint), .mlxs_coef_names(fit))
  expect_equal(colnames(sum_boot_ci$confint), c("5 %", "95 %"))
  expect_output(print(sum_boot_ci), "5 %")
  tidy_boot <- tidy(fit, bootstrap = TRUE, bootstrap_args = list(B = 15, seed = 123, progress = FALSE))
  expect_true(all(!is.na(tidy_boot$std.error)))

  sum_resid <- summary(
    fit,
    bootstrap = TRUE,
    bootstrap_args = list(
      bootstrap_type = "residual",
      B = 10,
      seed = 321,
      progress = FALSE
    )
  )
  expect_true(!is.null(sum_resid$bootstrap))
  expect_equal(length(sum_resid$bootstrap$se), length(drop(as.matrix(coef(fit)))))

  sum_ci <- summary(fit, confint = TRUE, level = 0.9)
  expect_equal(
    sum_ci$confint,
    confint(fit, level = 0.9),
    tolerance = 1e-6,
    ignore_attr = TRUE
  )
})

test_that("confint.mlxs_lm supports bootstrap percentile intervals", {
  fit <- mlxs_lm(mpg ~ cyl + disp, data = mtcars)
  args <- list(B = 12, seed = 123, progress = FALSE)

  ci <- confint(fit, bootstrap = TRUE, bootstrap_args = args)
  expect_equal(dim(ci), c(3L, 2L))
  expect_equal(rownames(ci), .mlxs_coef_names(fit))
  expect_equal(colnames(ci), c("2.5 %", "97.5 %"))
  expect_true(all(is.finite(ci)))
  expect_true(all(ci[, 1L] <= ci[, 2L]))

  ci_again <- confint(fit, bootstrap = TRUE, bootstrap_args = args)
  expect_equal(ci_again, ci)

  ci_cyl <- confint(
    fit,
    parm = "cyl",
    bootstrap = TRUE,
    bootstrap_args = args
  )
  expect_equal(dim(ci_cyl), c(1L, 2L))
  expect_equal(rownames(ci_cyl), "cyl")

  ci_resid <- confint(
    fit,
    parm = 2L,
    bootstrap = TRUE,
    bootstrap_args = list(
      bootstrap_type = "residual",
      B = 10,
      seed = 321,
      progress = FALSE
    )
  )
  expect_equal(dim(ci_resid), c(1L, 2L))
  expect_equal(rownames(ci_resid), "cyl")
})
