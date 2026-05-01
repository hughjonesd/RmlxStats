fuzz_tier <- skip_fuzz_tests("mlxs_glmnet")
skip_if_not_installed("glmnet")

summarise_glmnet_fit <- function(
  scenario,
  family,
  case,
  fit_pair,
  alpha,
  lambda_indices
) {
  lambda <- fit_pair$lambda
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
        case_type = "deterministic",
        scenario = scenario,
        family = family,
        n = nrow(case$x),
        p = ncol(case$x),
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

test_that("mlxs_glmnet deterministic fuzz cases match glmnet", {
  specs <- data.frame(
    scenario = c(
      "ar1_correlated", "ar1_correlated", "block_correlated",
      "null_signal", "null_signal", "strong_rare_binomial"
    ),
    family = c(
      "gaussian", "binomial", "gaussian",
      "gaussian", "binomial", "binomial"
    ),
    seed = c(1001L, 1002L, 1003L, 1004L, 1005L, 1006L),
    n = c(900L, 900L, 900L, 900L, 900L, 1200L),
    p = c(120L, 120L, 120L, 120L, 120L, 160L),
    rho = c(0.9, 0.9, 0, 0.8, 0.8, 0.5),
    alpha = c(1, 1, 0.5, 1, 1, 0.5)
  )

  summaries <- vector("list", nrow(specs))
  for (spec_idx in seq_len(nrow(specs))) {
    spec <- specs[spec_idx, ]
    case <- glmnet_fuzz_case(
      seed = spec$seed,
      scenario = spec$scenario,
      family = spec$family,
      n = spec$n,
      p = spec$p,
      n_test = 700L,
      rho = spec$rho
    )
    fit_pair <- fit_glmnet_pair(
      case,
      family = spec$family,
      alpha = spec$alpha,
      nlambda = 20L,
      lambda_min_ratio = 1e-3
    )
    lambda_indices <- unique(pmax(
      1L,
      pmin(length(fit_pair$lambda), c(1L, 10L, length(fit_pair$lambda)))
    ))
    summaries[[spec_idx]] <- summarise_glmnet_fit(
      scenario = spec$scenario,
      family = spec$family,
      case = case,
      fit_pair = fit_pair,
      alpha = spec$alpha,
      lambda_indices = lambda_indices
    )
  }
  summaries_df <- do.call(rbind, summaries)

  print(summaries_df, digits = 4)
  write_fuzz_summaries(
    summaries_df,
    suite = "mlxs-glmnet-deterministic",
    tier = fuzz_tier
  )

  expect_true(all(summaries_df$all_finite))
  expect_true(all(summaries_df$max_prediction_error <= 1e-4))
  expect_true(all(summaries_df$max_objective_error <= 1e-5))
})

test_that("mlxs_glmnet deterministic metamorphic properties hold", {
  case <- glmnet_fuzz_case(
    seed = 2001L,
    scenario = "ar1_correlated",
    family = "gaussian",
    n = 800L,
    p = 80L,
    n_test = 300L,
    rho = 0.85
  )
  fit_pair <- fit_glmnet_pair(
    case,
    family = "gaussian",
    alpha = 1,
    nlambda = 18L
  )
  lambda <- fit_pair$lambda
  fit <- fit_pair$mlx

  set.seed(2718)
  row_perm <- sample(seq_len(nrow(case$x)))
  perm_fit <- mlxs_glmnet(
    case$x[row_perm, , drop = FALSE],
    case$y[row_perm],
    family = mlxs_gaussian(),
    alpha = 1,
    lambda = lambda,
    standardize = FALSE,
    maxit = 5000L,
    tol = 1e-7
  )
  expect_equal(
    predict(perm_fit, newx = case$x_test, s = lambda),
    predict(fit, newx = case$x_test, s = lambda),
    tolerance = 1e-6,
    ignore_attr = TRUE
  )

  col_perm <- sample(seq_len(ncol(case$x)))
  col_fit <- mlxs_glmnet(
    case$x[, col_perm, drop = FALSE],
    case$y,
    family = mlxs_gaussian(),
    alpha = 1,
    lambda = lambda,
    standardize = FALSE,
    maxit = 5000L,
    tol = 1e-7
  )
  expect_equal(
    predict(col_fit, newx = case$x_test[, col_perm, drop = FALSE], s = lambda),
    predict(fit, newx = case$x_test, s = lambda),
    tolerance = 1e-6,
    ignore_attr = TRUE
  )
})
