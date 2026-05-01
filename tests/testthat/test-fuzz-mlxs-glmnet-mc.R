fuzz_tier <- skip_fuzz_tests("mlxs_glmnet")
skip_if_not_installed("glmnet")

run_glmnet_selection_rep <- function(
  seed,
  scenario,
  family,
  n_train,
  p,
  alpha,
  nlambda
) {
  case <- glmnet_fuzz_case(
    seed = seed,
    scenario = scenario,
    family = family,
    n = n_train,
    p = p,
    n_test = 10L,
    rho = 0.8
  )
  mlx_family <- if (family == "gaussian") {
    mlxs_gaussian()
  } else {
    mlxs_binomial()
  }
  fit <- mlxs_glmnet(
    case$x,
    case$y,
    family = mlx_family,
    alpha = alpha,
    nlambda = nlambda,
    lambda_min_ratio = 1e-3,
    standardize = FALSE,
    intercept = TRUE,
    maxit = 5000L,
    tol = 1e-7
  )

  truth_active <- abs(case$beta) > 1e-7
  n_active <- sum(truth_active)
  beta_path <- as.matrix(fit$beta)
  support_rows <- vector("list", ncol(beta_path))
  exact_recovery <- logical(ncol(beta_path))
  for (lambda_idx in seq_len(ncol(beta_path))) {
    support <- glmnet_fuzz_support(beta_path[, lambda_idx], case$beta)
    selected <- abs(beta_path[, lambda_idx]) > 1e-7
    exact_recovery[[lambda_idx]] <- isTRUE(all(selected == truth_active))
    support_rows[[lambda_idx]] <- data.frame(
      lambda_index = lambda_idx,
      lambda = fit$lambda[[lambda_idx]],
      exact_recovery = exact_recovery[[lambda_idx]],
      support,
      row.names = NULL
    )
  }
  support_df <- do.call(rbind, support_rows)
  # Use the true support only to summarize whether the fitted path contains a
  # good sparse model. This is not a practical tuning rule.
  best_idx <- order(
    abs(support_df$active_size - n_active),
    -support_df$true_positives,
    support_df$false_positives
  )[[1]]

  data.frame(
    scenario = scenario,
    family = family,
    n = n_train,
    p = p,
    alpha = alpha,
    lambda_index = support_df$lambda_index[[best_idx]],
    lambda = support_df$lambda[[best_idx]],
    selection_recovery_probability =
      as.numeric(support_df$exact_recovery[[best_idx]]),
    active_size = support_df$active_size[[best_idx]],
    true_positives = support_df$true_positives[[best_idx]],
    false_positives = support_df$false_positives[[best_idx]],
    false_negatives = support_df$false_negatives[[best_idx]],
    support_precision = support_df$support_precision[[best_idx]],
    support_recall = support_df$support_recall[[best_idx]],
    all_finite = all(is.finite(c(fit$a0, beta_path))),
    row.names = NULL
  )
}

summarise_glmnet_selection_mc <- function(results, reps) {
  numeric_cols <- vapply(results, is.numeric, logical(1))
  means <- colMeans(results[numeric_cols], na.rm = TRUE)
  data.frame(
    case_type = "monte_carlo",
    scenario = results$scenario[[1]],
    family = results$family[[1]],
    n = results$n[[1]],
    p = results$p[[1]],
    nreps = reps,
    alpha = results$alpha[[1]],
    lambda_index = means[["lambda_index"]],
    lambda = means[["lambda"]],
    selection_recovery_probability =
      means[["selection_recovery_probability"]],
    active_size = means[["active_size"]],
    true_positives = means[["true_positives"]],
    false_positives = means[["false_positives"]],
    false_negatives = means[["false_negatives"]],
    support_precision = means[["support_precision"]],
    support_recall = means[["support_recall"]],
    all_finite = all(results$all_finite),
    row.names = NULL
  )
}

run_glmnet_selection_mc <- function(
  scenario,
  family,
  seed0,
  reps,
  n_train,
  p,
  alpha,
  nlambda
) {
  set.seed(seed0)
  rep_seeds <- sample.int(.Machine$integer.max, reps)
  results <- vector("list", reps)
  for (rep_idx in seq_len(reps)) {
    results[[rep_idx]] <- tryCatch(
      run_glmnet_selection_rep(
        seed = rep_seeds[[rep_idx]],
        scenario = scenario,
        family = family,
        n_train = n_train,
        p = p,
        alpha = alpha,
        nlambda = nlambda
      ),
      error = function(err) {
        stop(
          "run_glmnet_selection_mc failed for scenario='",
          scenario,
          "', family='",
          family,
          "', rep=",
          rep_idx,
          ", seed=",
          rep_seeds[[rep_idx]],
          ". Reproduce with run_glmnet_selection_rep(seed = ",
          rep_seeds[[rep_idx]],
          ", scenario = '",
          scenario,
          "', family = '",
          family,
          "'): ",
          conditionMessage(err),
          call. = FALSE
        )
      }
    )
  }
  summarise_glmnet_selection_mc(do.call(rbind, results), reps = reps)
}

test_that("mlxs_glmnet Monte Carlo selection recovery is stable", {
  reps <- if (identical(fuzz_tier, "full")) 20L else 5L
  n_train <- if (identical(fuzz_tier, "full")) 1500L else 700L
  p <- if (identical(fuzz_tier, "full")) 500L else 180L
  nlambda <- if (identical(fuzz_tier, "full")) 20L else 12L

  summaries_df <- run_glmnet_selection_mc(
    scenario = "ar1_correlated",
    family = "gaussian",
    seed0 = 10000L,
    reps = reps,
    n_train = n_train,
    p = p,
    alpha = 1,
    nlambda = nlambda
  )

  print(summaries_df, digits = 4)
  write_fuzz_summaries(
    summaries_df,
    suite = "mlxs-glmnet-monte-carlo",
    tier = fuzz_tier
  )

  expect_true(all(summaries_df$all_finite))
  expect_true(summaries_df$support_recall >= 0.75)
  expect_true(summaries_df$selection_recovery_probability >= 0.2)
})
