fuzz_tier <- skip_fuzz_tests("mlxs_glmnet")
skip_if_not_installed("glmnet")

run_glmnet_mc_rep <- function(
  seed,
  scenario,
  family,
  n_train,
  n_test,
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
    n_test = n_test,
    rho = if (scenario == "ar1_correlated") 0.9 else 0.8
  )
  fit_pair <- fit_glmnet_pair(
    case,
    family = family,
    alpha = alpha,
    nlambda = nlambda,
    lambda_min_ratio = 1e-3
  )
  lambda <- fit_pair$lambda
  lambda_indices <- unique(pmax(
    1L,
    pmin(length(lambda), c(1L, ceiling(length(lambda) / 2), length(lambda)))
  ))
  mlx_pred <- predict(
    fit_pair$mlx,
    newx = case$x_test,
    s = lambda[lambda_indices],
    type = "response"
  )
  ref_pred <- predict(
    fit_pair$ref,
    newx = case$x_test,
    s = lambda[lambda_indices],
    type = "response"
  )
  mlx_beta <- as.matrix(fit_pair$mlx$beta)[, lambda_indices, drop = FALSE]
  ref_beta <- as.matrix(fit_pair$ref$beta)[, lambda_indices, drop = FALSE]
  mlx_a0 <- as.numeric(fit_pair$mlx$a0)[lambda_indices]
  ref_a0 <- as.numeric(fit_pair$ref$a0)[lambda_indices]

  rows <- vector("list", length(lambda_indices))
  for (row_idx in seq_along(lambda_indices)) {
    idx <- lambda_indices[[row_idx]]
    beta <- mlx_beta[, row_idx]
    ref_beta_col <- ref_beta[, row_idx]
    obj <- glmnet_fuzz_objective(
      case$x,
      case$y,
      beta,
      mlx_a0[[row_idx]],
      lambda[[idx]],
      alpha,
      family = family
    )
    ref_obj <- glmnet_fuzz_objective(
      case$x,
      case$y,
      ref_beta_col,
      ref_a0[[row_idx]],
      lambda[[idx]],
      alpha,
      family = family
    )
    test_loss <- glmnet_fuzz_loss(
      case$y_test,
      mlx_pred[, row_idx],
      family = family
    )
    ref_loss <- glmnet_fuzz_loss(
      case$y_test,
      ref_pred[, row_idx],
      family = family
    )
    oracle_loss <- glmnet_fuzz_loss(
      case$y_test,
      case$oracle_test_pred,
      family = family
    )
    support <- glmnet_fuzz_support(beta, case$beta)
    rows[[row_idx]] <- cbind(
      data.frame(
        scenario = scenario,
        family = family,
        n = n_train,
        p = p,
        alpha = alpha,
        lambda_index = idx,
        lambda = lambda[[idx]],
        bias = mean(beta - case$beta),
        rmse = sqrt(mean((beta - case$beta)^2)),
        test_loss = test_loss,
        reference_test_loss = ref_loss,
        oracle_test_loss = oracle_loss,
        excess_risk = test_loss - oracle_loss,
        loss_error = abs(test_loss - ref_loss),
        relative_loss_error = abs(test_loss - ref_loss) /
          max(abs(ref_loss), 1e-12),
        max_coef_error = max(abs(beta - ref_beta_col)),
        max_prediction_error = max(abs(mlx_pred[, row_idx] -
          ref_pred[, row_idx])),
        max_objective_error = abs(obj - ref_obj),
        all_finite = all(is.finite(c(
          beta, mlx_a0[[row_idx]], mlx_pred[, row_idx], test_loss, obj
        ))),
        row.names = NULL
      ),
      support
    )
  }
  do.call(rbind, rows)
}

summarise_glmnet_mc <- function(results, reps) {
  groups <- split(
    results,
    interaction(
      results$scenario,
      results$family,
      results$lambda_index,
      drop = TRUE
    )
  )
  rows <- lapply(groups, function(group) {
    numeric_cols <- vapply(group, is.numeric, logical(1))
    means <- colMeans(group[numeric_cols], na.rm = TRUE)
    data.frame(
      case_type = "monte_carlo",
      scenario = group$scenario[[1]],
      family = group$family[[1]],
      n = group$n[[1]],
      p = group$p[[1]],
      nreps = reps,
      alpha = group$alpha[[1]],
      lambda_index = group$lambda_index[[1]],
      lambda = means[["lambda"]],
      bias = means[["bias"]],
      rmse = means[["rmse"]],
      test_loss = means[["test_loss"]],
      reference_test_loss = means[["reference_test_loss"]],
      oracle_test_loss = means[["oracle_test_loss"]],
      excess_risk = means[["excess_risk"]],
      loss_error = means[["loss_error"]],
      relative_loss_error = means[["relative_loss_error"]],
      active_size = means[["active_size"]],
      true_positives = means[["true_positives"]],
      false_positives = means[["false_positives"]],
      false_negatives = means[["false_negatives"]],
      support_precision = means[["support_precision"]],
      support_recall = means[["support_recall"]],
      max_coef_error = max(group$max_coef_error),
      max_prediction_error = max(group$max_prediction_error),
      max_objective_error = max(group$max_objective_error),
      all_finite = all(group$all_finite),
      row.names = NULL
    )
  })
  do.call(rbind, rows)
}

run_glmnet_mc <- function(
  scenario,
  family,
  seed0,
  reps,
  n_train,
  n_test,
  p,
  alpha,
  nlambda
) {
  set.seed(seed0)
  rep_seeds <- sample.int(.Machine$integer.max, reps)
  results <- vector("list", reps)
  for (rep_idx in seq_len(reps)) {
    results[[rep_idx]] <- tryCatch(
      run_glmnet_mc_rep(
        seed = rep_seeds[[rep_idx]],
        scenario = scenario,
        family = family,
        n_train = n_train,
        n_test = n_test,
        p = p,
        alpha = alpha,
        nlambda = nlambda
      ),
      error = function(err) {
        stop(
          "run_glmnet_mc failed for scenario='",
          scenario,
          "', family='",
          family,
          "', rep=",
          rep_idx,
          ", seed=",
          rep_seeds[[rep_idx]],
          ". Reproduce with run_glmnet_mc_rep(seed = ",
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
  summarise_glmnet_mc(do.call(rbind, results), reps = reps)
}

test_that("mlxs_glmnet Monte Carlo fuzz summaries are within tolerance", {
  reps <- if (identical(fuzz_tier, "full")) 500L else 100L
  n_train <- if (identical(fuzz_tier, "full")) 2500L else 1000L
  n_test <- if (identical(fuzz_tier, "full")) 2500L else 1000L
  p <- if (identical(fuzz_tier, "full")) 1000L else 300L
  nlambda <- if (identical(fuzz_tier, "full")) 50L else 30L

  specs <- data.frame(
    scenario = c(
      "ar1_correlated", "block_correlated", "near_null",
      "strong_rare_binomial"
    ),
    family = c("gaussian", "gaussian", "gaussian", "binomial"),
    seed0 = c(10000L, 20000L, 30000L, 40000L),
    alpha = c(1, 0.5, 1, 0.5)
  )

  summaries <- vector("list", nrow(specs))
  for (spec_idx in seq_len(nrow(specs))) {
    spec <- specs[spec_idx, ]
    summaries[[spec_idx]] <- run_glmnet_mc(
      scenario = spec$scenario,
      family = spec$family,
      seed0 = spec$seed0,
      reps = reps,
      n_train = n_train,
      n_test = n_test,
      p = p,
      alpha = spec$alpha,
      nlambda = nlambda
    )
  }
  summaries_df <- do.call(rbind, summaries)

  print(summaries_df, digits = 4)
  write_fuzz_summaries(
    summaries_df,
    suite = "mlxs-glmnet-monte-carlo",
    tier = fuzz_tier
  )

  expect_true(all(summaries_df$all_finite))
  expect_true(all(summaries_df$max_prediction_error <= 1e-4))
  expect_true(all(summaries_df$max_objective_error <= 1e-5))
  expect_true(all(summaries_df$relative_loss_error <= 1e-3))
})
