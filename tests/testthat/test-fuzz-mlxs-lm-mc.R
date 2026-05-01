fuzz_tier <- skip_fuzz_tests("mlxs_lm")

summarise_lm_mc <- function(results, truth, scenario, reps) {
  estimates <- mc_field_matrix(results, "estimates", names(truth))
  ses <- mc_field_matrix(results, "ses", names(truth))
  covered <- mc_field_matrix(results, "covered", names(truth))
  stopifnot(!anyNA(covered))
  bias <- colMeans(estimates) - unname(truth)
  empirical_se <- apply(estimates, 2, sd)
  coverage <- colMeans(covered)
  metric_names <- c(
    "truth", "estimate", "bias", "error", "standard_error",
    "standard_error", "coverage"
  )
  fuzz_metric_rows(
    list(
      case_type = "monte_carlo",
      scenario = scenario,
      nreps = reps,
      n = 80L,
      p = length(truth)
    ),
    term = rep(names(truth), each = length(metric_names)),
    measure = rep(metric_names, times = length(truth)),
    target      = c("coefficient", "coefficient", "coefficient", "coefficient", "coefficient", "coefficient", "confidence_interval"),
    source      = c("truth",       "mlx",         "mlx",         "mlx",         "empirical",   "model",       "mlx"),
    baseline    = c(NA,            "truth",       "truth",       "truth",       NA,            "empirical",   "truth"),
    aggregation = c("value",       "mean",        "mean",        "rmse",        "value",       "mean",        "mean"),
    value = c(rbind(
      unname(truth),
      colMeans(estimates),
      bias,
      sqrt(colMeans((sweep(estimates, 2, unname(truth)))^2)),
      empirical_se,
      colMeans(ses),
      coverage
    )),
    value_se = c(rbind(
      rep(NA_real_, length(truth)),
      rep(NA_real_, length(truth)),
      empirical_se / sqrt(reps),
      rep(NA_real_, length(truth)),
      rep(NA_real_, length(truth)),
      rep(NA_real_, length(truth)),
      sqrt(coverage * (1 - coverage) / reps)
    ))
  )
}

run_lm_mc_rep <- function(
  seed,
  scenario = c("homoskedastic", "heteroskedastic"),
  truth = c("(Intercept)" = 1, x1 = 0.75, x2 = -0.5, x3 = 0.25)
) {
  scenario <- match.arg(scenario)
  set.seed(seed)
  n <- 80
  x <- make_design(n = n, p = 3, rho = 0.35)
  colnames(x) <- names(truth)[-1L]

  # Heteroscedastic errors deliberately break the usual OLS standard-error
  # model, so this scenario is reported but not used for nominal coverage.
  sigma <- if (scenario == "heteroskedastic") {
    0.45 * (1 + abs(x[, 1]))
  } else {
    rep(0.7, n)
  }
  y <- truth[1] + drop(x %*% truth[-1L]) + rnorm(n, sd = sigma)
  data <- data.frame(y = y, x)
  fit <- mlxs_lm(y ~ x1 + x2 + x3, data = data)
  ci <- confint(fit)
  sum_fit <- summary(fit)

  list(
    estimates = coef_vector(fit),
    ses = as.numeric(sum_fit$std.error),
    covered = ci[, 1] <= truth & truth <= ci[, 2]
  )
}

run_lm_bootstrap_mc_rep <- function(
  seed,
  bootstrap_B,
  scenario = c("skew_homoskedastic", "heavy_tail_homoskedastic"),
  truth = c("(Intercept)" = 0.2, x1 = 0.5, x2 = -0.3, x3 = 0.2, x4 = 0.1),
  n = 20000L
) {
  scenario <- match.arg(scenario)
  set.seed(seed)
  x <- make_design(n = n, p = length(truth) - 1L, rho = 0.3)
  colnames(x) <- names(truth)[-1L]

  # These errors have mean zero and variance one, but are not normal. The
  # bootstrap check asks whether resampling recovers the empirical coefficient
  # variation under a large-n non-Gaussian DGP.
  err <- if (scenario == "skew_homoskedastic") {
    (exp(rnorm(n)) - exp(0.5)) / sqrt((exp(1) - 1) * exp(1))
  } else {
    rt(n, df = 5) / sqrt(5 / 3)
  }
  y <- truth[1] + drop(x %*% truth[-1L]) + err
  data <- data.frame(y = y, x)
  fit <- mlxs_lm(y ~ x1 + x2 + x3 + x4, data = data)
  sum_fit <- summary(fit)
  boot_sum <- summary(
    fit,
    bootstrap = TRUE,
    bootstrap_args = list(B = bootstrap_B, seed = seed, progress = FALSE)
  )

  list(
    estimates = coef_vector(fit),
    ses = as.numeric(sum_fit$std.error),
    boot_ses = as.numeric(boot_sum$std.error)
  )
}

summarise_lm_bootstrap_mc <- function(
  results,
  truth,
  scenario,
  reps,
  bootstrap_B,
  n
) {
  estimates <- mc_field_matrix(results, "estimates", names(truth))
  ses <- mc_field_matrix(results, "ses", names(truth))
  boot_ses <- mc_field_matrix(results, "boot_ses", names(truth))

  empirical_se <- apply(estimates, 2, sd, na.rm = TRUE)
  average_model_se <- colMeans(ses, na.rm = TRUE)
  average_bootstrap_se <- colMeans(boot_ses, na.rm = TRUE)
  all_finite <- vapply(seq_along(truth), function(idx) {
    vals <- c(estimates[, idx], ses[, idx], boot_ses[, idx])
    all(is.finite(vals[!is.na(vals)]))
  }, logical(1))
  coef_metrics <- c(
    "truth", "estimate", "bias", "standard_error", "standard_error",
    "standard_error", "standard_error_ratio", "standard_error_ratio",
    "diagnostic"
  )
  rbind(
    fuzz_metric_rows(
      list(
        case_type = "monte_carlo",
        scenario = scenario,
        n = n,
        p = length(truth),
        nreps = reps,
        bootstrap_B = bootstrap_B
      ),
      term = rep(names(truth), each = length(coef_metrics)),
      measure = rep(coef_metrics, times = length(truth)),
      target      = c("coefficient", "coefficient", "coefficient", "coefficient", "coefficient", "coefficient", "coefficient", "coefficient", "finite"),
      source      = c("truth",       "mlx",         "mlx",         "empirical",   "model",       "bootstrap",  "model",       "bootstrap",  "mlx"),
      baseline    = c(NA,            "truth",       "truth",       NA,            "empirical",   "empirical",  "empirical",   "empirical",  NA),
      aggregation = c("value",       "mean",        "mean",        "value",       "mean",        "mean",       "ratio",       "ratio",      "all"),
      value = c(rbind(
        unname(truth),
        colMeans(estimates, na.rm = TRUE),
        colMeans(estimates, na.rm = TRUE) - unname(truth),
        empirical_se,
        average_model_se,
        average_bootstrap_se,
        average_model_se / empirical_se,
        average_bootstrap_se / empirical_se,
        as.numeric(all_finite)
      ))
    ),
    fuzz_metric_rows(
      list(
        case_type = "monte_carlo",
        scenario = scenario,
        n = n,
        p = length(truth),
        nreps = reps,
        bootstrap_B = bootstrap_B
      ),
      measure = "diagnostic",
      target = "bootstrap_failure",
      source = "bootstrap",
      aggregation = "rate",
      value = 0
    )
  )
}

test_that("mlxs_lm Monte Carlo fuzz summaries are within tolerance", {
  hom_reps <- if (identical(fuzz_tier, "full")) 10000L else 2000L
  het_reps <- if (identical(fuzz_tier, "full")) 2000L else 500L
  truth <- c("(Intercept)" = 1, x1 = 0.75, x2 = -0.5, x3 = 0.25)
  hom_results <- run_mc_reps(
    reps = hom_reps,
    seed0 = 10000,
    rep_fun = run_lm_mc_rep,
    label = "run_lm_mc",
    truth = truth,
    scenario = "homoskedastic"
  )
  het_results <- run_mc_reps(
    reps = het_reps,
    seed0 = 20000,
    rep_fun = run_lm_mc_rep,
    label = "run_lm_mc",
    truth = truth,
    scenario = "heteroskedastic"
  )
  hom <- summarise_lm_mc(hom_results, truth, "homoskedastic", hom_reps)
  het <- summarise_lm_mc(het_results, truth, "heteroskedastic", het_reps)
  summaries_df <- rbind(hom, het)
  write_fuzz_summaries(
    summaries_df,
    suite = "mlxs-lm-monte-carlo",
    tier = fuzz_tier
  )

  # The report notes that estimating 95% coverage to about +/- 0.01
  # needs roughly 1,825 replications. The fast tier uses 2,000
  # homoskedastic replications for a meaningful coverage screen; the full
  # tier uses 10,000. The 4-MCSE band should almost never fail by
  # simulation noise alone, while still catching large regressions.
  hom_bias <- hom[hom$measure == "bias", ]
  expect_true(
    all(abs(hom_bias$value) <= 4 * hom_bias$value_se),
    info = paste(
      "homoskedastic bias outside Monte Carlo band:",
      paste(hom_bias$term[abs(hom_bias$value) > 4 * hom_bias$value_se],
            collapse = ", ")
    )
  )

  het_bias <- het[het$measure == "bias", ]
  expect_true(
    all(abs(het_bias$value) <= 4 * het_bias$value_se),
    info = paste(
      "heteroskedastic bias outside Monte Carlo band:",
      paste(het_bias$term[abs(het_bias$value) > 4 * het_bias$value_se],
            collapse = ", ")
    )
  )

  hom_coverage <- hom[hom$measure == "coverage", ]
  coverage_band <- 4 * hom_coverage$value_se
  expect_true(
    all(abs(hom_coverage$value - 0.95) <= coverage_band),
    info = paste(
      "homoskedastic coverage outside Monte Carlo band:",
      paste(hom_coverage$term[
        abs(hom_coverage$value - 0.95) > coverage_band
      ], collapse = ", ")
    )
  )
})

test_that("mlxs_lm bootstrap SE calibration is stable", {
  reps <- if (identical(fuzz_tier, "full")) 300L else 120L
  n <- if (identical(fuzz_tier, "full")) 50000L else 20000L
  bootstrap_B <- if (identical(fuzz_tier, "full")) 100L else 50L
  scenarios <- c(
    skew_homoskedastic = 30000L,
    heavy_tail_homoskedastic = 40000L
  )
  truth <- c("(Intercept)" = 0.2, x1 = 0.5, x2 = -0.3, x3 = 0.2, x4 = 0.1)
  summaries <- vector("list", length(scenarios))
  names(summaries) <- names(scenarios)
  for (scenario in names(scenarios)) {
    results <- run_mc_reps(
      reps = reps,
      seed0 = scenarios[[scenario]],
      rep_fun = run_lm_bootstrap_mc_rep,
      label = "run_lm_bootstrap_mc",
      bootstrap_B = bootstrap_B,
      scenario = scenario,
      truth = truth,
      n = n
    )
    summaries[[scenario]] <- summarise_lm_bootstrap_mc(
      results = results,
      truth = truth,
      scenario = scenario,
      reps = reps,
      bootstrap_B = bootstrap_B,
      n = n
    )
  }
  summaries_df <- do.call(rbind, summaries)
  write_fuzz_summaries(
    summaries_df,
    suite = "mlxs-lm-monte-carlo",
    tier = fuzz_tier
  )

  lower <- if (identical(fuzz_tier, "full")) 0.88 else 0.80
  upper <- if (identical(fuzz_tier, "full")) 1.15 else 1.25
  failure <- summaries_df[
    summaries_df$target == "bootstrap_failure" &
      summaries_df$aggregation == "rate",
  ]
  finite <- summaries_df[
    summaries_df$target == "finite" & summaries_df$aggregation == "all",
  ]
  boot_ratio <- summaries_df[
    summaries_df$measure == "standard_error_ratio" &
      summaries_df$source == "bootstrap",
  ]
  expect_true(all(failure$value == 0))
  expect_true(all(as.logical(finite$value)))
  expect_true(
    all(boot_ratio$value >= lower & boot_ratio$value <= upper),
    info = paste(
      "bootstrap SE ratio outside calibration band:",
      paste(boot_ratio$term[
        boot_ratio$value < lower | boot_ratio$value > upper
      ], collapse = ", ")
    )
  )
})
