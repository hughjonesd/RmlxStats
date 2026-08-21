fuzz_tier <- skip_fuzz_tests("mlxs_glmnet")
skip_if_not_installed("glmnet")

run_glmnet_prediction_rep <- function(
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
    rho = 0.8
  )
  ref <- glmnet::glmnet(
    case$x,
    case$y,
    family = family,
    alpha = alpha,
    nlambda = nlambda,
    lambda.min.ratio = 1e-3,
    standardize = FALSE,
    intercept = TRUE,
    control = list(thresh = 1e-12, maxit = 100000L)
  )
  lambda <- as.numeric(ref$lambda)

  # Compare predictions with the true conditional mean, not noisy test outcomes.
  # This keeps the MC target focused on estimator/algorithm variation.
  true_mean_risk <- function(pred) {
    mean((as.numeric(pred) - case$oracle_test_pred)^2)
  }

  ref_pred <- predict(
    ref,
    newx = case$x_test,
    s = lambda,
    type = "response"
  )
  ref_risk <- apply(ref_pred, 2, true_mean_risk)
  best_idx <- which.min(ref_risk)

  mlx_family <- if (family == "gaussian") {
    mlxs_gaussian()
  } else {
    mlxs_binomial()
  }
  lambda_used <- lambda[seq_len(best_idx)]
  mlx <- mlxs_glmnet(
    case$x,
    case$y,
    family = mlx_family,
    alpha = alpha,
    lambda = lambda_used,
    standardize = FALSE,
    intercept = TRUE,
    maxit = 5000L,
    tol = 1e-7
  )
  mlx_pred <- predict(
    mlx,
    newx = case$x_test,
    s = lambda[[best_idx]],
    type = "response"
  )
  mlx_risk <- true_mean_risk(mlx_pred)

  data.frame(
    scenario = scenario,
    family = family,
    n = n_train,
    p = p,
    alpha = alpha,
    lambda_index = best_idx,
    lambda = lambda[[best_idx]],
    mlx_prediction_risk = mlx_risk,
    reference_prediction_risk = ref_risk[[best_idx]],
    prediction_risk_delta = mlx_risk - ref_risk[[best_idx]],
    all_finite = all(is.finite(c(
      as.numeric(mlx$a0),
      as.matrix(mlx$beta),
      mlx_risk,
      ref_risk
    ))),
    row.names = NULL
  )
}

summarise_glmnet_prediction_mc <- function(results, reps) {
  mcse <- function(x) sd(x, na.rm = TRUE) / sqrt(reps)
  means <- colMeans(results[vapply(results, is.numeric, logical(1))],
                    na.rm = TRUE)
  fuzz_metric_rows(
    list(
      case_type = "monte_carlo",
      scenario = results$scenario[[1]],
      family = results$family[[1]],
      n = results$n[[1]],
      p = results$p[[1]],
      nreps = reps,
      alpha = results$alpha[[1]],
      lambda_index = means[["lambda_index"]],
      lambda = means[["lambda"]]
    ),
    measure     = c("loss",       "loss",       "delta", "diagnostic"),
    target      = c("prediction", "prediction", "risk",  "finite"),
    source      = c("mlx",        "reference",  "mlx",   "mlx"),
    baseline    = c("truth",      "truth",      "reference", NA),
    aggregation = c("mean",       "mean",       "mean",  "all"),
    value = c(
      means[["mlx_prediction_risk"]],
      means[["reference_prediction_risk"]],
      means[["prediction_risk_delta"]],
      as.numeric(all(results$all_finite))
    ),
    value_se = c(
      mcse(results$mlx_prediction_risk),
      mcse(results$reference_prediction_risk),
      mcse(results$prediction_risk_delta),
      NA_real_
    )
  )
}

test_that("mlxs_glmnet Monte Carlo prediction risk is stable", {
  skip_if(
    !identical(fuzz_tier, "full"),
    "glmnet Monte Carlo prediction risk runs only in the full tier."
  )

  reps <- 50L
  n_train <- 1500L
  n_test <- 5000L
  p <- 500L
  nlambda <- 20L

  results <- run_mc_reps(
    reps = reps,
    seed0 = 10000L,
    rep_fun = run_glmnet_prediction_rep,
    label = "run_glmnet_prediction_mc",
    scenario = "ar1_correlated",
    family = "gaussian",
    n_train = n_train,
    n_test = n_test,
    p = p,
    alpha = 1,
    nlambda = nlambda
  )
  summaries_df <- summarise_glmnet_prediction_mc(
    do.call(rbind, results),
    reps = reps
  )
  write_fuzz_summaries(
    summaries_df,
    suite = "mlxs-glmnet-monte-carlo",
    tier = fuzz_tier
  )

  finite <- summaries_df[summaries_df$target == "finite", ]
  risk_delta <- summaries_df[
    summaries_df$measure == "delta" & summaries_df$target == "risk",
  ]
  expect_true(all(as.logical(finite$value)))
  expect_true(
    risk_delta$value <= 0.02,
    info = paste("prediction risk delta:", signif(risk_delta$value, 4))
  )
})
