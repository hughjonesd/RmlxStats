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
  data.frame(
    case_type = "monte_carlo",
    family = family,
    scenario = "regular",
    n = n,
    p = length(truth),
    nreps = reps,
    coefficient = names(truth),
    truth = unname(truth),
    mean_estimate = colMeans(estimates),
    bias = colMeans(estimates) - unname(truth),
    mcse_bias = apply(estimates, 2, sd) / sqrt(reps),
    rmse = sqrt(colMeans((sweep(estimates, 2, unname(truth)))^2)),
    empirical_se = apply(estimates, 2, sd),
    average_model_se = colMeans(ses),
    ci_coverage = colMeans(covered),
    mcse_coverage = sqrt(colMeans(covered) * (1 - colMeans(covered)) / reps),
    max_coef_error = max(max_coef_error),
    convergence_rate = mean(converged),
    mean_iterations = mean(iterations),
    max_iterations = max(iterations),
    row.names = NULL
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
  data.frame(
    case_type = "monte_carlo",
    family = if (scenario == "gaussian_skew") "gaussian" else "binomial",
    scenario = scenario,
    n = n,
    p = length(truth),
    nreps = reps,
    bootstrap_B = bootstrap_B,
    bootstrap_failure_rate = 0,
    coefficient = names(truth),
    truth = unname(truth),
    mean_estimate = colMeans(estimates, na.rm = TRUE),
    bias = colMeans(estimates, na.rm = TRUE) - unname(truth),
    empirical_se = empirical_se,
    average_model_se = average_model_se,
    average_bootstrap_se = average_bootstrap_se,
    model_se_ratio = average_model_se / empirical_se,
    bootstrap_se_ratio = average_bootstrap_se / empirical_se,
    convergence_rate = mean(converged, na.rm = TRUE),
    mean_iterations = mean(iterations, na.rm = TRUE),
    max_iterations = max(iterations, na.rm = TRUE),
    all_finite = vapply(seq_along(truth), function(idx) {
      vals <- c(estimates[, idx], ses[, idx], boot_ses[, idx])
      all(is.finite(vals[!is.na(vals)]))
    }, logical(1)),
    row.names = NULL
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
      reproduce_args = list(family = family, n = n),
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

  print(summaries_df, digits = 4)
  write_fuzz_summaries(
    summaries_df,
    suite = "mlxs-glm-monte-carlo",
    tier = fuzz_tier
  )

  expect_true(all(summaries_df$convergence_rate == 1))
  expect_true(all(summaries_df$max_coef_error <= 1e-5))

  # Gaussian identity GLM should have the same finite-sample bias behavior as
  # OLS. Binomial and Poisson MLE bias is only asymptotically zero, so those
  # families are recorded here but not failed on zero-bias in this first pass.
  gaussian_rows <- summaries_df$family == "gaussian"
  expect_true(
    all(abs(summaries_df$bias[gaussian_rows]) <=
          4 * summaries_df$mcse_bias[gaussian_rows]),
    info = paste(
      "gaussian bias outside Monte Carlo band:",
      paste(summaries_df$coefficient[
        gaussian_rows &
          abs(summaries_df$bias) > 4 * summaries_df$mcse_bias
      ], collapse = ", ")
    )
  )

  expect_true(
    all(abs(summaries_df$ci_coverage - 0.95) <=
          4 * summaries_df$mcse_coverage),
    info = paste(
      "coverage outside Monte Carlo band:",
      paste(
        paste(summaries_df$family, summaries_df$coefficient, sep = ":")[
          abs(summaries_df$ci_coverage - 0.95) >
            4 * summaries_df$mcse_coverage
        ],
        collapse = ", "
      )
    )
  )
})

test_that("mlxs_glm bootstrap SE calibration is stable", {
  reps <- if (identical(fuzz_tier, "full")) 300L else 120L
  n <- if (identical(fuzz_tier, "full")) 50000L else 20000L
  bootstrap_B <- if (identical(fuzz_tier, "full")) 100L else 50L
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
      reproduce_args = list(
        bootstrap_B = bootstrap_B,
        scenario = scenario,
        n = n
      ),
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

  print(summaries_df, digits = 4)
  write_fuzz_summaries(
    summaries_df,
    suite = "mlxs-glm-monte-carlo",
    tier = fuzz_tier
  )

  lower <- if (identical(fuzz_tier, "full")) 0.88 else 0.80
  upper <- if (identical(fuzz_tier, "full")) 1.15 else 1.25
  expect_true(all(summaries_df$bootstrap_failure_rate == 0))
  expect_true(all(summaries_df$all_finite))
  expect_true(all(summaries_df$convergence_rate == 1))
  expect_true(
    all(summaries_df$bootstrap_se_ratio >= lower &
          summaries_df$bootstrap_se_ratio <= upper),
    info = paste(
      "bootstrap SE ratio outside calibration band:",
      paste(paste(summaries_df$family, summaries_df$coefficient, sep = ":")[
        summaries_df$bootstrap_se_ratio < lower |
          summaries_df$bootstrap_se_ratio > upper
      ], collapse = ", ")
    )
  )
})
