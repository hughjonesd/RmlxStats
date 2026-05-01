fuzz_tier <- skip_fuzz_tests("mlxs_glm")

glm_beta <- function(p, beta_scale = 1) {
  beta <- numeric(p)
  active <- seq_len(min(8L, p))
  beta[active] <- beta_scale * seq(0.55, by = -0.1, length.out = length(active))
  beta
}

compare_glm_to_stats_glm <- function(
  scenario,
  family,
  formula,
  data,
  weights = NULL
) {
  families <- glm_family_pair(family)
  if (is.null(weights)) {
    base_fit <- glm(formula, data = data, family = families$base)
    mlx_fit <- mlxs_glm(formula, data = data, family = families$mlx)
  } else {
    data$.weights <- weights
    base_fit <- glm(
      formula,
      data = data,
      family = families$base,
      weights = .weights
    )
    mlx_fit <- mlxs_glm(
      formula,
      data = data,
      family = families$mlx,
      weights = .weights
    )
  }
  mlx_coef <- coef_vector(mlx_fit)
  mlx_vcov <- as.matrix(vcov(mlx_fit))
  mlx_fitted <- drop(as.matrix(fitted(mlx_fit)))
  mlx_eta <- drop(as.matrix(predict(mlx_fit, type = "link")))
  finite_values <- c(mlx_coef, mlx_vcov, mlx_fitted, mlx_eta)

  fuzz_metric_rows(
    list(
      case_type = "deterministic",
      family = family,
      scenario = scenario,
      n = nrow(model.frame(formula, data)),
      p = ncol(model.matrix(formula, data))
    ),
    measure     = c("diagnostic",       "error",       "error",     "error",            "error", "error",    "error", "diagnostic",  "diagnostic", "diagnostic"),
    target      = c("condition_number", "coefficient", "fitted",    "linear_predictor", "vcov",  "deviance", "aic",   "convergence", "iterations", "finite"),
    source      = c("design",           "mlx",         "mlx",       "mlx",              "mlx",   "mlx",      "mlx",   "mlx",         "mlx",        "mlx"),
    baseline    = c(NA,                 "reference",   "reference", "reference",        "reference", "reference", "reference", NA,  NA,           NA),
    aggregation = c("value",            "max",         "max",       "max",              "max",   "value",    "value", "value",       "value",      "all"),
    value = c(
      kappa(model.matrix(formula, data)),
      max(abs(mlx_coef - coef(base_fit))),
      max(abs(mlx_fitted - fitted(base_fit))),
      max(abs(mlx_eta - predict(base_fit, type = "link"))),
      max(abs(mlx_vcov - vcov(base_fit))),
      abs(mlx_fit$deviance - base_fit$deviance),
      abs(mlx_fit$aic - base_fit$aic),
      as.numeric(mlx_fit$converged),
      mlx_fit$iter,
      as.numeric(all(is.finite(finite_values)))
    )
  )
}

expect_mlxs_glm_matches_glm <- function(
  formula,
  data,
  family,
  weights = NULL,
  coef_tol = 1e-6,
  value_tol = 1e-6,
  label = ""
) {
  families <- glm_family_pair(family)
  if (is.null(weights)) {
    base_fit <- glm(formula, data = data, family = families$base)
    mlx_fit <- mlxs_glm(formula, data = data, family = families$mlx)
  } else {
    data$.weights <- weights
    base_fit <- glm(
      formula,
      data = data,
      family = families$base,
      weights = .weights
    )
    mlx_fit <- mlxs_glm(
      formula,
      data = data,
      family = families$mlx,
      weights = .weights
    )
  }

  expect_true(mlx_fit$converged, info = label)
  expect_equal(coef_vector(mlx_fit), coef(base_fit),
               tolerance = coef_tol, ignore_attr = TRUE, info = label)
  expect_equal(drop(as.matrix(fitted(mlx_fit))), fitted(base_fit),
               tolerance = value_tol, ignore_attr = TRUE, info = label)
  expect_equal(drop(as.matrix(predict(mlx_fit, type = "link"))),
               predict(base_fit, type = "link"),
               tolerance = value_tol, ignore_attr = TRUE, info = label)
  expect_equal(as.matrix(vcov(mlx_fit)), vcov(base_fit),
               tolerance = coef_tol, ignore_attr = TRUE, info = label)
  expect_equal(mlx_fit$deviance, base_fit$deviance,
               tolerance = value_tol, info = label)
  expect_equal(mlx_fit$aic, base_fit$aic,
               tolerance = value_tol, info = label)

  list(base = base_fit, mlx = mlx_fit)
}

test_that("mlxs_glm deterministic differential fuzz cases match stats::glm", {
  specs <- data.frame(
    family = c(
      "gaussian", "binomial", "poisson",
      "gaussian", "binomial", "poisson",
      "gaussian", "binomial", "poisson",
      "binomial", "poisson",
      "binomial", "poisson"
    ),
    scenario = c(
      "regular_correlated", "regular_correlated", "regular_correlated",
      "large_n", "large_n", "large_n",
      "weighted", "rare_event", "overdispersed",
      "large_p", "large_p",
      "large_p_rare_event", "large_p_overdispersed"
    ),
    seed = c(
      1L, 2L, 3L,
      101L, 102L, 103L,
      151L, 152L, 153L,
      201L, 202L,
      251L, 252L
    ),
    n = c(
      1000L, 1000L, 1000L,
      100000L, 100000L, 100000L,
      2000L, 2000L, 2000L,
      15000L, 15000L,
      15000L, 15000L
    ),
    p = c(8L, 8L, 8L, 20L, 12L, 12L, 8L, 8L, 8L, 150L, 150L, 150L, 150L),
    rho = c(
      0.45, 0.45, 0.45,
      0.2, 0.2, 0.2,
      0.3, 0.3, 0.3,
      0.35, 0.35,
      0.35, 0.35
    )
  )
  if (identical(fuzz_tier, "full")) {
    specs <- rbind(
      specs,
      data.frame(
        family = c("binomial", "poisson"),
        scenario = c("larger_p", "larger_p"),
        seed = c(301L, 302L),
        n = c(30000L, 30000L),
        p = c(300L, 300L),
        rho = c(0.25, 0.25)
      )
    )
  }

  summaries <- vector("list", nrow(specs))
  for (idx in seq_len(nrow(specs))) {
    spec <- specs[idx, ]
    beta_scale <- if (grepl("large_p|larger_p", spec$scenario)) 0.35 else 1
    case <- if (grepl("overdispersed", spec$scenario)) {
      make_case(
        seed = spec$seed,
        family = "poisson",
        n = spec$n,
        p = spec$p,
        rho = spec$rho,
        intercept = -0.15,
        beta = glm_beta(spec$p, beta_scale),
        poisson_overdispersion = 2
      )
    } else {
      make_case(
        seed = spec$seed,
        family = spec$family,
        n = spec$n,
        p = spec$p,
        rho = spec$rho,
        noise = 0.7,
        intercept = if (grepl("rare_event", spec$scenario)) -2.5 else -0.15,
        beta = glm_beta(spec$p, beta_scale)
      )
    }
    weights <- if (identical(spec$scenario, "weighted")) {
      seq(0.5, 1.5, length.out = nrow(case$data))
    } else {
      NULL
    }
    label <- paste(spec$family, spec$scenario, sep = ":")
    expect_mlxs_glm_matches_glm(
      case$formula,
      case$data,
      family = spec$family,
      weights = weights,
      coef_tol = if (grepl("overdispersed", spec$scenario)) 1e-4 else 1e-5,
      value_tol = 1e-5,
      label = label
    )
    summaries[[idx]] <- compare_glm_to_stats_glm(
      scenario = spec$scenario,
      family = spec$family,
      formula = case$formula,
      data = case$data,
      weights = weights
    )
  }

  summaries_df <- do.call(rbind, summaries)
  write_fuzz_summaries(
    summaries_df,
    suite = "mlxs-glm-deterministic",
    tier = fuzz_tier
  )
  converged <- summaries_df[
    summaries_df$target == "convergence" &
      summaries_df$aggregation == "value",
  ]
  finite <- summaries_df[
    summaries_df$target == "finite" & summaries_df$aggregation == "all",
  ]
  coef_error <- summaries_df[
    summaries_df$target == "coefficient" &
      summaries_df$aggregation == "max",
  ]
  vcov_error <- summaries_df[
    summaries_df$target == "vcov" & summaries_df$aggregation == "max",
  ]
  expect_true(all(as.logical(converged$value)))
  expect_true(all(as.logical(finite$value)))
  expect_true(all(coef_error$value <= 1e-5))
  expect_true(all(vcov_error$value <= 1e-5))
})

test_that("mlxs_glm metamorphic fuzz properties hold", {
  for (family in c("gaussian", "binomial", "poisson")) {
    case <- make_case(
      seed = 17,
      family = family,
      n = 1200,
      p = 4,
      rho = 0.4,
      noise = 0.7,
      intercept = -0.15,
      beta = glm_beta(4)
    )
    data <- case$data
    families <- glm_family_pair(family)
    fit <- mlxs_glm(case$formula, data = data, family = families$mlx)
    refit <- mlxs_glm(case$formula, data = data, family = families$mlx)
    expect_equal(coef_vector(refit), coef_vector(fit),
                 tolerance = 1e-12, ignore_attr = TRUE, info = family)

    set.seed(2718)
    perm <- sample(seq_len(nrow(data)))
    perm_fit <- mlxs_glm(
      case$formula,
      data = data[perm, ],
      family = families$mlx
    )
    expect_equal(coef_vector(perm_fit), coef_vector(fit),
                 tolerance = 1e-6, ignore_attr = TRUE, info = family)
    expect_equal(drop(as.matrix(fitted(perm_fit)))[order(perm)],
                 drop(as.matrix(fitted(fit))),
                 tolerance = 1e-6, ignore_attr = TRUE, info = family)

    col_fit <- mlxs_glm(
      y ~ x4 + x2 + x1 + x3,
      data = data,
      family = families$mlx
    )
    expect_equal(coef_vector(col_fit)[names(coef_vector(fit))],
                 coef_vector(fit),
                 tolerance = 1e-6, ignore_attr = TRUE, info = family)
    expect_equal(drop(as.matrix(predict(col_fit, newdata = data))),
                 drop(as.matrix(predict(fit, newdata = data))),
                 tolerance = 1e-6, ignore_attr = TRUE, info = family)

    expect_equal(drop(as.matrix(predict(fit, type = "response"))),
                 drop(as.matrix(predict(
                   fit,
                   newdata = data,
                   type = "response"
                 ))),
                 tolerance = 1e-6, ignore_attr = TRUE, info = family)
  }

  gaussian <- make_case(
    seed = 99,
    family = "gaussian",
    n = 2000,
    p = 6,
    noise = 0.7,
    intercept = -0.15,
    beta = glm_beta(6)
  )
  glm_fit <- mlxs_glm(gaussian$formula, data = gaussian$data,
                      family = mlxs_gaussian())
  lm_fit <- mlxs_lm(gaussian$formula, data = gaussian$data)
  expect_equal(coef_vector(glm_fit), coef_vector(lm_fit),
               tolerance = 1e-6, ignore_attr = TRUE)
  expect_equal(drop(as.matrix(fitted(glm_fit))),
               drop(as.matrix(fitted(lm_fit))),
               tolerance = 1e-6, ignore_attr = TRUE)
  expect_equal(as.matrix(vcov(glm_fit)), as.matrix(vcov(lm_fit)),
               tolerance = 1e-6, ignore_attr = TRUE)
})
