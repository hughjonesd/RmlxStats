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
  z <- qnorm(0.975)
  ci <- cbind(estimates - z * ses, estimates + z * ses)

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
  estimates,
  ses,
  covered,
  max_coef_error,
  converged,
  iterations,
  truth,
  family,
  reps,
  n
) {
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

run_glm_mc <- function(
  reps,
  seed0,
  family = c("gaussian", "binomial", "poisson"),
  n = 1000L
) {
  family <- match.arg(family)
  truth <- c("(Intercept)" = -0.15, x1 = 0.55, x2 = 0.35, x3 = 0.15, x4 = -0.05)
  estimates <- matrix(NA_real_, nrow = reps, ncol = length(truth))
  ses <- matrix(NA_real_, nrow = reps, ncol = length(truth))
  covered <- matrix(NA, nrow = reps, ncol = length(truth))
  max_coef_error <- rep(NA_real_, reps)
  converged <- rep(NA, reps)
  iterations <- rep(NA_integer_, reps)
  colnames(estimates) <- colnames(ses) <- colnames(covered) <- names(truth)

  set.seed(seed0)
  rep_seeds <- sample.int(.Machine$integer.max, reps)
  for (rep_idx in seq_len(reps)) {
    rep_result <- tryCatch(
      run_glm_mc_rep(
        seed = rep_seeds[rep_idx],
        family = family,
        truth = truth,
        n = n
      ),
      error = function(err) {
        stop(
          "run_glm_mc failed for family='",
          family,
          "', rep=",
          rep_idx,
          ", seed=",
          rep_seeds[rep_idx],
          ". Reproduce with run_glm_mc_rep(seed = ",
          rep_seeds[rep_idx],
          ", family = '",
          family,
          "'): ",
          conditionMessage(err),
          call. = FALSE
        )
      }
    )
    estimates[rep_idx, ] <- rep_result$estimates
    ses[rep_idx, ] <- rep_result$ses
    covered[rep_idx, ] <- rep_result$covered
    max_coef_error[rep_idx] <- rep_result$max_coef_error
    converged[rep_idx] <- rep_result$converged
    iterations[rep_idx] <- rep_result$iterations
  }

  stopifnot(!anyNA(covered))

  summarise_glm_mc(
    estimates = estimates,
    ses = ses,
    covered = covered,
    max_coef_error = max_coef_error,
    converged = converged,
    iterations = iterations,
    truth = truth,
    family = family,
    reps = reps,
    n = n
  )
}

test_that("mlxs_glm Monte Carlo fuzz summaries are within tolerance", {
  reps <- if (identical(fuzz_tier, "full")) 2000L else 500L
  n <- if (identical(fuzz_tier, "full")) 2500L else 1000L
  mc_seeds <- c(gaussian = 10000L, binomial = 20000L, poisson = 30000L)
  summaries <- vector("list", length(mc_seeds))
  names(summaries) <- names(mc_seeds)
  for (family in names(mc_seeds)) {
    summaries[[family]] <- run_glm_mc(
      reps = reps,
      seed0 = mc_seeds[[family]],
      family = family,
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
