fuzz_tier <- skip_fuzz_tests("mlxs_lm")

summarise_lm_mc <- function(estimates, ses, covered, truth, scenario, reps) {
  out <- data.frame(
    case_type = "monte_carlo",
    scenario = scenario,
    coefficient = names(truth),
    truth = unname(truth),
    mean_estimate = colMeans(estimates),
    bias = colMeans(estimates) - unname(truth),
    mcse_bias = apply(estimates, 2, sd) / sqrt(reps),
    rmse = sqrt(colMeans((sweep(
      estimates, 2, unname(truth)
    ))^2)),
    empirical_se = apply(estimates, 2, sd),
    average_model_se = colMeans(ses),
    ci_coverage = colMeans(covered),
    mcse_coverage = sqrt(colMeans(covered) * (1 - colMeans(covered)) / reps),
    nreps = reps,
    n = 80L,
    p = length(truth),
    row.names = NULL
  )
  out
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

run_lm_mc <- function(
  reps,
  seed0,
  scenario = c("homoskedastic", "heteroskedastic")
) {
  scenario <- match.arg(scenario)
  truth <- c("(Intercept)" = 1, x1 = 0.75, x2 = -0.5, x3 = 0.25)
  estimates <- matrix(NA_real_, nrow = reps, ncol = length(truth))
  ses <- matrix(NA_real_, nrow = reps, ncol = length(truth))
  covered <- matrix(NA, nrow = reps, ncol = length(truth))
  colnames(estimates) <- colnames(ses) <- colnames(covered) <- names(truth)

  set.seed(seed0)
  rep_seeds <- sample.int(.Machine$integer.max, reps)
  for (rep_idx in seq_len(reps)) {
    rep_result <- tryCatch(
      run_lm_mc_rep(
        seed = rep_seeds[rep_idx],
        scenario = scenario,
        truth = truth
      ),
      error = function(err) {
        stop(
          "run_lm_mc failed for scenario='",
          scenario,
          "', rep=",
          rep_idx,
          ", seed=",
          rep_seeds[rep_idx],
          ". Reproduce with run_lm_mc_rep(seed = ",
          rep_seeds[rep_idx],
          ", scenario = '",
          scenario,
          "'): ",
          conditionMessage(err),
          call. = FALSE
        )
      }
    )
    estimates[rep_idx, ] <- rep_result$estimates
    ses[rep_idx, ] <- rep_result$ses
    covered[rep_idx, ] <- rep_result$covered
  }

  stopifnot(!anyNA(covered))

  summarise_lm_mc(estimates, ses, covered, truth, scenario, reps)
}

test_that("mlxs_lm Monte Carlo fuzz summaries are within tolerance", {
  hom_reps <- if (identical(fuzz_tier, "full")) 10000L else 2000L
  het_reps <- if (identical(fuzz_tier, "full")) 2000L else 500L
  hom <- run_lm_mc(
    reps = hom_reps,
    seed0 = 10000,
    scenario = "homoskedastic"
  )
  het <- run_lm_mc(
    reps = het_reps,
    seed0 = 20000,
    scenario = "heteroskedastic"
  )
  summaries_df <- rbind(hom, het)

  print(summaries_df, digits = 4)
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
  hom_bias_mcse <- hom$empirical_se / sqrt(hom_reps)
  expect_true(
    all(abs(hom$bias) <= 4 * hom_bias_mcse),
    info = paste(
      "homoskedastic bias outside Monte Carlo band:",
      paste(hom$coefficient[abs(hom$bias) > 4 * hom_bias_mcse],
            collapse = ", ")
    )
  )

  het_bias_mcse <- het$empirical_se / sqrt(het_reps)
  expect_true(
    all(abs(het$bias) <= 4 * het_bias_mcse),
    info = paste(
      "heteroskedastic bias outside Monte Carlo band:",
      paste(het$coefficient[abs(het$bias) > 4 * het_bias_mcse],
            collapse = ", ")
    )
  )

  coverage_band <- 4 * hom$mcse_coverage
  expect_true(
    all(abs(hom$ci_coverage - 0.95) <= coverage_band),
    info = paste(
      "homoskedastic coverage outside Monte Carlo band:",
      paste(hom$coefficient[
        abs(hom$ci_coverage - 0.95) > coverage_band
      ], collapse = ", ")
    )
  )
})
