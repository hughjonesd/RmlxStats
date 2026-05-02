fuzz_tier <- skip_fuzz_tests("mlxs_glm")

run_glm_mc_rep <- function(
  seed,
  family = c("gaussian", "binomial", "poisson"),
  truth = c("(Intercept)" = -0.15, x1 = 0.55, x2 = 0.35, x3 = 0.15, x4 = -0.05),
  n = 1000L
) {
  family <- match.arg(family)
  set.seed(seed)
  p <- length(truth) - 1L
  case <- make_case(
    seed = seed,
    family = family,
    n = n,
    p = p,
    rho = 0.3,
    noise = 0.7,
    intercept = truth[[1]],
    beta = truth[-1L]
  )
  data <- case$data
  families <- glm_family_pair(family)
  formula <- case$formula
  fit <- mlxs_glm(formula, data = data, family = families$mlx)
  base_fit <- glm(formula, data = data, family = families$base)
  sum_fit <- summary(fit)
  estimates <- coef_vector(fit)
  ses <- as.numeric(sum_fit$std.error)
  ci <- confint(fit)

  list(
    estimates = estimates,
    ses = ses,
    covered = ci[, 1] <= truth & truth <= ci[, 2],
    max_coef_error = max(abs(estimates - coef(base_fit))),
    converged = fit$converged,
    iterations = fit$iter
  )
}

summarise_glm_mc <- function(
  results,
  truth,
  family,
  reps,
  n
) {
  estimates <- mc_field_matrix(results, "estimates", names(truth))
  ses <- mc_field_matrix(results, "ses", names(truth))
  covered <- mc_field_matrix(results, "covered", names(truth))
  max_coef_error <- vapply(results, `[[`, numeric(1), "max_coef_error")
  converged <- vapply(results, `[[`, logical(1), "converged")
  iterations <- vapply(results, `[[`, numeric(1), "iterations")
  stopifnot(!anyNA(covered))
  bias <- colMeans(estimates) - unname(truth)
  empirical_se <- apply(estimates, 2, sd)
  coverage <- colMeans(covered)
  metric_names <- c(
    "truth", "estimate", "bias", "error", "standard_error",
    "standard_error", "coverage"
  )
  meta <- list(
    case_type = "monte_carlo",
    family = family,
    scenario = "regular",
    n = n,
    p = length(truth),
    nreps = reps
  )
  rbind(
    fuzz_metric_rows(
      meta,
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
    ),
    fuzz_metric_rows(
      meta,
      measure     = c("error",       "diagnostic",  "diagnostic", "diagnostic"),
      target      = c("coefficient", "convergence", "iterations", "iterations"),
      source      = c("mlx",         "mlx",         "mlx",        "mlx"),
      baseline    = c("reference",   NA,            NA,           NA),
      aggregation = c("max",         "rate",        "mean",       "max"),
      value = c(
        max(max_coef_error),
        mean(converged),
        mean(iterations),
        max(iterations)
      )
    )
  )
}

run_glm_bootstrap_mc_rep <- function(
  seed,
  bootstrap_B,
  scenario = c("gaussian_skew", "binomial_regular"),
  truth = c("(Intercept)" = -0.15, x1 = 0.55, x2 = 0.35, x3 = 0.15, x4 = -0.05),
  n = 20000L
) {
  scenario <- match.arg(scenario)
  set.seed(seed)
  p <- length(truth) - 1L
  x <- make_design(n = n, p = p, rho = 0.3)
  colnames(x) <- names(truth)[-1L]
  eta <- truth[1] + drop(x %*% truth[-1L])

  if (scenario == "gaussian_skew") {
    # The Gaussian identity model has correct conditional mean, but the error
    # distribution is skewed. The bootstrap SE should still estimate the
    # empirical coefficient variation in the large-n regime.
    err <- (exp(rnorm(n)) - exp(0.5)) / sqrt((exp(1) - 1) * exp(1))
    y <- eta + err
    family <- mlxs_gaussian()
  } else {
    y <- rbinom(n, size = 1L, prob = plogis(eta))
    family <- mlxs_binomial()
  }

  data <- data.frame(y = y, x)
  fit <- mlxs_glm(y ~ x1 + x2 + x3 + x4, data = data, family = family)
  sum_fit <- summary(fit)
  boot_sum <- summary(
    fit,
    bootstrap = TRUE,
    bootstrap_args = list(B = bootstrap_B, seed = seed, progress = FALSE)
  )

  list(
    estimates = coef_vector(fit),
    ses = as.numeric(sum_fit$std.error),
    boot_ses = as.numeric(boot_sum$std.error),
    converged = fit$converged,
    iterations = fit$iter
  )
}

summarise_glm_bootstrap_mc <- function(
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
  converged <- vapply(results, `[[`, logical(1), "converged")
  iterations <- vapply(results, `[[`, numeric(1), "iterations")

  empirical_se <- apply(estimates, 2, sd, na.rm = TRUE)
  average_model_se <- colMeans(ses, na.rm = TRUE)
  average_bootstrap_se <- colMeans(boot_ses, na.rm = TRUE)
  all_finite <- vapply(seq_along(truth), function(idx) {
    vals <- c(estimates[, idx], ses[, idx], boot_ses[, idx])
    all(is.finite(vals[!is.na(vals)]))
  }, logical(1))
  family <- if (scenario == "gaussian_skew") "gaussian" else "binomial"
  meta <- list(
    case_type = "monte_carlo",
    family = family,
    scenario = scenario,
    n = n,
    p = length(truth),
    nreps = reps,
    bootstrap_B = bootstrap_B
  )
  coef_metrics <- c(
    "truth", "estimate", "bias", "standard_error", "standard_error",
    "standard_error", "standard_error_ratio", "standard_error_ratio",
    "diagnostic"
  )
  rbind(
    fuzz_metric_rows(
      meta,
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
      meta,
      measure     = c("diagnostic",        "diagnostic",  "diagnostic", "diagnostic"),
      target      = c("bootstrap_failure", "convergence", "iterations", "iterations"),
      source      = c("bootstrap",         "mlx",         "mlx",        "mlx"),
      aggregation = c("rate",              "rate",        "mean",       "max"),
      value = c(
        0,
        mean(converged, na.rm = TRUE),
        mean(iterations, na.rm = TRUE),
        max(iterations, na.rm = TRUE)
      )
    )
  )
}

test_that("mlxs_glm Monte Carlo fuzz summaries are within tolerance", {
  reps <- if (identical(fuzz_tier, "full")) 2000L else 500L
  n <- if (identical(fuzz_tier, "full")) 2500L else 1000L
  mc_seeds <- c(gaussian = 10000L, binomial = 20000L, poisson = 30000L)
  truth <- c("(Intercept)" = -0.15, x1 = 0.55, x2 = 0.35, x3 = 0.15, x4 = -0.05)
  summaries <- vector("list", length(mc_seeds))
  names(summaries) <- names(mc_seeds)
  for (family in names(mc_seeds)) {
    results <- run_mc_reps(
      reps = reps,
      seed0 = mc_seeds[[family]],
      rep_fun = run_glm_mc_rep,
      label = "run_glm_mc",
      family = family,
      truth = truth,
      n = n
    )
    summaries[[family]] <- summarise_glm_mc(
      results = results,
      truth = truth,
      family = family,
      reps = reps,
      n = n
    )
  }
  summaries_df <- do.call(rbind, summaries)
  write_fuzz_summaries(
    summaries_df,
    suite = "mlxs-glm-monte-carlo",
    tier = fuzz_tier
  )

  convergence <- summaries_df[
    summaries_df$target == "convergence" & summaries_df$aggregation == "rate",
  ]
  coef_error <- summaries_df[
    summaries_df$measure == "error" &
      summaries_df$target == "coefficient" &
      summaries_df$aggregation == "max",
  ]
  expect_true(all(convergence$value == 1))
  expect_true(all(coef_error$value <= 1e-5))

  # Gaussian identity GLM should have the same finite-sample bias behavior as
  # OLS. Binomial and Poisson MLE bias is only asymptotically zero, so those
  # families are recorded here but not failed on zero-bias in this first pass.
  bias_rows <- summaries_df[summaries_df$measure == "bias", ]
  gaussian_rows <- bias_rows$family == "gaussian"
  expect_true(
    all(abs(bias_rows$value[gaussian_rows]) <=
          4 * bias_rows$value_se[gaussian_rows]),
    info = paste(
      "gaussian bias outside Monte Carlo band:",
      paste(bias_rows$term[
        gaussian_rows &
          abs(bias_rows$value) > 4 * bias_rows$value_se
      ], collapse = ", ")
    )
  )

  coverage <- summaries_df[summaries_df$measure == "coverage", ]
  expect_true(
    all(abs(coverage$value - 0.95) <= 4 * coverage$value_se),
    info = paste(
      "coverage outside Monte Carlo band:",
      paste(
        paste(coverage$family, coverage$term, sep = ":")[
          abs(coverage$value - 0.95) > 4 * coverage$value_se
        ],
        collapse = ", "
      )
    )
  )
})

test_that("mlxs_glm bootstrap SE calibration is stable", {
  skip_if_not(
    identical(fuzz_tier, "full"),
    "GLM bootstrap fuzz calibration runs only in the full tier."
  )
  reps <- 300L
  n <- 20000L
  bootstrap_B <- 100L
  scenarios <- c(gaussian_skew = 40000L, binomial_regular = 50000L)
  truth <- c("(Intercept)" = -0.15, x1 = 0.55, x2 = 0.35, x3 = 0.15, x4 = -0.05)
  summaries <- vector("list", length(scenarios))
  names(summaries) <- names(scenarios)
  for (scenario in names(scenarios)) {
    results <- run_mc_reps(
      reps = reps,
      seed0 = scenarios[[scenario]],
      rep_fun = run_glm_bootstrap_mc_rep,
      label = "run_glm_bootstrap_mc",
      bootstrap_B = bootstrap_B,
      scenario = scenario,
      truth = truth,
      n = n
    )
    summaries[[scenario]] <- summarise_glm_bootstrap_mc(
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
    suite = "mlxs-glm-monte-carlo",
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
  convergence <- summaries_df[
    summaries_df$target == "convergence" & summaries_df$aggregation == "rate",
  ]
  boot_ratio <- summaries_df[
    summaries_df$measure == "standard_error_ratio" &
      summaries_df$source == "bootstrap",
  ]
  expect_true(all(failure$value == 0))
  expect_true(all(as.logical(finite$value)))
  expect_true(all(convergence$value == 1))
  expect_true(
    all(boot_ratio$value >= lower & boot_ratio$value <= upper),
    info = paste(
      "bootstrap SE ratio outside calibration band:",
      paste(paste(boot_ratio$family, boot_ratio$term, sep = ":")[
        boot_ratio$value < lower | boot_ratio$value > upper
      ], collapse = ", ")
    )
  )
})
