fuzz_tier <- skip_fuzz_tests("mlxs_lm")

expect_mlxs_lm_matches_lm <- function(
  formula,
  data,
  weights = NULL,
  coef_tol = 1e-5,
  value_tol = 1e-6,
  label = ""
) {
  base_fit <- if (is.null(weights)) {
    lm(formula, data = data)
  } else {
    data$.weights <- weights
    lm(formula, data = data, weights = .weights)
  }
  mlx_fit <- if (is.null(weights)) {
    mlxs_lm(formula, data = data)
  } else {
    data$.weights <- weights
    mlxs_lm(formula, data = data, weights = .weights)
  }

  expect_equal(coef_vector(mlx_fit), coef(base_fit),
               tolerance = coef_tol, ignore_attr = TRUE, info = label)
  expect_equal(drop(as.matrix(fitted(mlx_fit))), fitted(base_fit),
               tolerance = value_tol, ignore_attr = TRUE, info = label)
  expect_equal(drop(as.matrix(residuals(mlx_fit))), residuals(base_fit),
               tolerance = value_tol, ignore_attr = TRUE, info = label)
  expect_equal(as.matrix(vcov(mlx_fit)), vcov(base_fit),
               tolerance = coef_tol, ignore_attr = TRUE, info = label)
  expect_equal(confint(mlx_fit), confint(base_fit),
               tolerance = coef_tol, ignore_attr = TRUE, info = label)
  expect_equal(mlx_fit$rank, base_fit$rank, info = label)
  expect_equal(mlx_fit$df.residual, base_fit$df.residual, info = label)

  list(base = base_fit, mlx = mlx_fit)
}

compare_lm_to_stats_lm <- function(scenario, formula, data, case_type) {
  mlx_fit <- mlxs_lm(formula, data = data)
  base_fit <- lm(formula, data = data)
  mlx_coef <- coef_vector(mlx_fit)
  mlx_vcov <- as.matrix(vcov(mlx_fit))
  mlx_fitted <- drop(as.matrix(fitted(mlx_fit)))
  finite_values <- c(mlx_coef, mlx_vcov, mlx_fitted)

  fuzz_metric_rows(
    list(
      case_type = case_type,
      scenario = scenario,
      n = nrow(model.frame(formula, data)),
      p = ncol(model.matrix(formula, data))
    ),
    measure     = c("diagnostic",       "error",       "error",     "error", "diagnostic"),
    target      = c("condition_number", "coefficient", "fitted",    "vcov",  "finite"),
    source      = c("design",           "mlx",         "mlx",       "mlx",   "mlx"),
    baseline    = c(NA,                 "reference",   "reference", "reference", NA),
    aggregation = c("value",            "max",         "max",       "max",   "all"),
    value = c(
      kappa(model.matrix(formula, data)),
      max(abs(mlx_coef - coef(base_fit))),
      max(abs(mlx_fitted - fitted(base_fit))),
      max(abs(mlx_vcov - vcov(base_fit))),
      as.numeric(all(is.finite(finite_values)))
    )
  )
}

test_that("mlxs_lm deterministic differential fuzz cases match stats::lm", {
  cases <- list(
    iid = make_case(seed = 1, n = 48, p = 4),
    ar1_high_rho = make_case(seed = 17, n = 72, p = 5, rho = 0.92),
    leverage = {
      obj <- make_case(seed = 2718, n = 64, p = 4)
      obj$data[1, c("x1", "x2")] <- c(8, -7)
      obj$data$y[1] <- obj$data$y[1] + 3
      obj
    },
    near_collinear = {
      set.seed(314159)
      n <- 80
      x1 <- rnorm(n)
      x2 <- x1 + rnorm(n, sd = 0.03)
      x3 <- rnorm(n)
      y <- 1 + 0.8 * x1 - 0.5 * x2 + 0.25 * x3 + rnorm(n, sd = 0.35)
      list(data = data.frame(y, x1, x2, x3), formula = y ~ x1 + x2 + x3)
    }
  )

  for (case_name in names(cases)) {
    case <- cases[[case_name]]
    expect_mlxs_lm_matches_lm(
      case$formula,
      case$data,
      coef_tol = 1e-5,
      value_tol = 1e-5,
      label = case_name
    )
  }

  weighted <- make_case(seed = 8675309, n = 60, p = 3)
  weights <- seq(0.25, 1.5, length.out = nrow(weighted$data))
  expect_mlxs_lm_matches_lm(
    weighted$formula,
    weighted$data,
    weights = weights,
    coef_tol = 1e-6,
    value_tol = 1e-6,
    label = "weighted"
  )

  specs <- data.frame(
    scenario = c("large_n", "large_p"),
    seed = c(101L, 102L),
    n = c(20000L, 10000L),
    p = c(30L, 300L)
  )
  if (identical(fuzz_tier, "full")) {
    specs <- rbind(
      specs,
      data.frame(scenario = "larger_p", seed = 103L, n = 30000L, p = 500L)
    )
  }

  summaries <- vector("list", nrow(specs))
  for (idx in seq_len(nrow(specs))) {
    spec <- specs[idx, ]
    set.seed(spec$seed)
    x <- make_design(n = spec$n, p = spec$p)
    beta <- numeric(spec$p)
    active <- seq_len(min(10L, spec$p))
    beta[active] <- seq(1, by = -0.12, length.out = length(active))
    y <- 0.5 + drop(x %*% beta) + rnorm(spec$n, sd = 0.25)
    data <- data.frame(y = y, x)
    names(data) <- c("y", paste0("x", seq_len(spec$p)))
    formula <- as.formula(
      paste("y ~", paste(names(data)[-1L], collapse = " + "))
    )

    expect_mlxs_lm_matches_lm(
      formula,
      data,
      coef_tol = 1e-5,
      value_tol = 1e-5,
      label = spec$scenario
    )
    summaries[[idx]] <- compare_lm_to_stats_lm(
      scenario = spec$scenario,
      formula = formula,
      data = data,
      case_type = "large_design"
    )
  }

  summaries_df <- do.call(rbind, summaries)
  write_fuzz_summaries(
    summaries_df,
    suite = "mlxs-lm-deterministic",
    tier = fuzz_tier
  )
})

test_that("mlxs_lm metamorphic fuzz properties hold", {
  case <- make_case(seed = 17, n = 70, p = 3, rho = 0.5)
  data <- case$data
  fit <- mlxs_lm(case$formula, data = data)

  set.seed(2718)
  perm <- sample(seq_len(nrow(data)))
  perm_fit <- mlxs_lm(case$formula, data = data[perm, ])
  expect_equal(coef_vector(perm_fit), coef_vector(fit),
               tolerance = 1e-6, ignore_attr = TRUE)
  expect_equal(drop(as.matrix(fitted(perm_fit)))[order(perm)],
               drop(as.matrix(fitted(fit))),
               tolerance = 1e-6, ignore_attr = TRUE)

  col_fit <- mlxs_lm(y ~ x3 + x1 + x2, data = data)
  expect_equal(coef_vector(col_fit)[names(coef_vector(fit))],
               coef_vector(fit),
               tolerance = 1e-6, ignore_attr = TRUE)
  expect_equal(drop(as.matrix(predict(col_fit, newdata = data))),
               drop(as.matrix(predict(fit, newdata = data))),
               tolerance = 1e-6, ignore_attr = TRUE)

  expect_equal(drop(as.matrix(predict(fit))),
               drop(as.matrix(predict(fit, newdata = data))),
               tolerance = 1e-6, ignore_attr = TRUE)
})

test_that("mlxs_lm rank-deficient fuzz cases fail clearly", {
  data <- make_case(seed = 1, n = 40, p = 3)$data
  data$x_dup <- data$x1
  expect_error(
    mlxs_lm(y ~ x1 + x2 + x_dup, data = data),
    "full-rank model matrix",
    fixed = TRUE
  )

  data$x_combo <- data$x1 + data$x2
  expect_error(
    mlxs_lm(y ~ x1 + x2 + x3 + x_combo, data = data),
    "full-rank model matrix",
    fixed = TRUE
  )
})

test_that("mlxs_lm near-rank-deficient stability is tracked", {
  set.seed(1717)
  n <- 160
  x1 <- rnorm(n)
  x3 <- rnorm(n)
  z <- rnorm(n)
  noise <- rnorm(n, sd = 0.2)
  eps_grid <- if (identical(fuzz_tier, "full")) {
    c(1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
  } else {
    c(1e-1, 1e-2, 1e-3, 1e-4)
  }

  summaries <- vector("list", length(eps_grid))
  for (idx in seq_along(eps_grid)) {
    eps <- eps_grid[[idx]]
    x2 <- x1 + eps * z
    y <- 1 + 0.8 * x1 - 0.5 * x2 + 0.25 * x3 + noise
    data <- data.frame(y, x1, x2, x3)
    scenario <- paste0("epsilon_", format(eps, scientific = TRUE))
    summaries[[idx]] <- compare_lm_to_stats_lm(
      scenario = scenario,
      formula = y ~ x1 + x2 + x3,
      data = data,
      case_type = "near_rank_deficient"
    )
  }

  summaries_df <- do.call(rbind, summaries)
  write_fuzz_summaries(
    summaries_df,
    suite = "mlxs-lm-deterministic",
    tier = fuzz_tier
  )

  finite <- summaries_df[
    summaries_df$target == "finite" & summaries_df$aggregation == "all",
  ]
  fitted_error <- summaries_df[
    summaries_df$target == "fitted" & summaries_df$aggregation == "max",
  ]
  coef_error <- summaries_df[
    summaries_df$target == "coefficient" & summaries_df$aggregation == "max",
  ]
  vcov_error <- summaries_df[
    summaries_df$target == "vcov" & summaries_df$aggregation == "max",
  ]
  condition <- summaries_df[summaries_df$target == "condition_number", ]
  expect_true(all(as.logical(finite$value)),
              info = "non-finite near-singular fit")
  expect_true(
    all(fitted_error$value <= 0.05),
    info = paste(fitted_error$scenario[fitted_error$value > 0.05],
                 collapse = ", ")
  )

  moderate_scenarios <- condition$scenario[condition$value <= 500]
  moderate <- coef_error$scenario %in% moderate_scenarios
  expect_true(
    all(coef_error$value[moderate] <= 1e-4),
    info = paste(coef_error$scenario[
      moderate & coef_error$value > 1e-4
    ], collapse = ", ")
  )
  moderate <- vcov_error$scenario %in% moderate_scenarios
  expect_true(
    all(vcov_error$value[moderate] <= 1e-5),
    info = paste(vcov_error$scenario[
      moderate & vcov_error$value > 1e-5
    ], collapse = ", ")
  )
})

test_that("mlxs_lm NIST StRD fixtures are checked", {
  # Norris is a lower-difficulty certified regression check where MLX should
  # match the published coefficients, SEs, sigma, and R-squared. Longley is a
  # classic multicollinearity stress case; Wampler1 is an exact polynomial fit
  # with severe scaling. Those harder cases are tracked as finite-output and
  # R-squared diagnostics rather than strict single-precision coefficient
  # or standard-error checks.
  source(
    testthat::test_path("fixtures", "nist-strd", "linear-regression.R"),
    local = TRUE
  )

  summaries <- vector("list", length(nist_lm_cases))
  names(summaries) <- names(nist_lm_cases)
  for (case_name in names(nist_lm_cases)) {
    case <- nist_lm_cases[[case_name]]
    fit <- mlxs_lm(case$formula, data = case$data)
    fit_summary <- summary(fit)
    coefs <- coef_vector(fit)
    se <- setNames(as.numeric(fit_summary$std.error), names(coefs))
    sigma <- fit_summary$sigma
    r_squared <- fit_summary$r.squared
    finite_values <- c(coefs, se, sigma, r_squared)

    summaries[[case_name]] <- fuzz_metric_rows(
      list(
        case_type = "nist_strd",
        scenario = case$name,
        n = nrow(model.frame(case$formula, case$data)),
        p = ncol(model.matrix(case$formula, case$data))
      ),
      measure     = c("diagnostic",       "error",       "error",          "error",          "error",     "diagnostic"),
      target      = c("condition_number", "coefficient", "standard_error", "residual_sigma", "r_squared", "finite"),
      source      = c("design",           "mlx",         "mlx",            "mlx",            "mlx",       "mlx"),
      baseline    = c(NA,                 "reference",   "reference",      "reference",      "reference", NA),
      aggregation = c("value",            "max",         "max",            "value",          "value",     "all"),
      value = c(
        kappa(model.matrix(case$formula, case$data)),
        max(abs(coefs[names(case$coef)] - case$coef)),
        max(abs(se[names(case$se)] - case$se)),
        abs(sigma - case$sigma),
        abs(r_squared - case$r_squared),
        as.numeric(all(is.finite(finite_values)))
      )
    )

    expect_true(all(is.finite(finite_values)), info = case$name)
    if (isTRUE(case$strict_check)) {
      expect_equal(coefs[names(case$coef)], case$coef,
                   tolerance = 1e-4, ignore_attr = TRUE, info = case$name)
      expect_equal(se[names(case$se)], case$se,
                   tolerance = 1e-5, ignore_attr = TRUE, info = case$name)
      expect_equal(sigma, case$sigma, tolerance = 1e-5, info = case$name)
      expect_equal(r_squared, case$r_squared,
                   tolerance = 1e-8, info = case$name)
    } else {
      expect_equal(r_squared, case$r_squared,
                   tolerance = 1e-6, info = case$name)
    }
  }

  summaries_df <- do.call(rbind, summaries)
  write_fuzz_summaries(
    summaries_df,
    suite = "mlxs-lm-deterministic",
    tier = fuzz_tier
  )
})
