fuzz_tier <- skip_fuzz_tests("RmlxStats benchmarks")

benchmark_specs <- function(tier) {
  if (identical(tier, "full")) {
    data.frame(
      method = c(
        "mlxs_lm", "mlxs_lm",
        "mlxs_lm_summary",
        "mlxs_lm_predict",
        "mlxs_glm", "mlxs_glm",
        "mlxs_glm_summary",
        "mlxs_glm_predict",
        "mlxs_glmnet_predict",
        "mlxs_cv_glmnet", "mlxs_cv_glmnet",
        "mlxs_prcomp", "mlxs_prcomp",
        "mlxs_prcomp_summary",
        "mlxs_prcomp_predict",
        "mlxs_lm_bootstrap_summary"
      ),
      scenario = c(
        "lm_large_n", "lm_large_p",
        "lm_summary_large_n",
        "lm_predict_large_n",
        "glm_large_n", "glm_large_p",
        "glm_summary_large_n",
        "glm_predict_large_n",
        "glmnet_predict_large_n",
        "cv_glmnet_large_n", "cv_glmnet_large_p",
        "prcomp_large_n", "prcomp_large_p",
        "prcomp_summary_large_p",
        "prcomp_predict_large_n",
        "lm_bootstrap_summary"
      ),
      n = c(
        180000L, 8500L,
        120000L,
        120000L,
        105000L, 8500L,
        80000L,
        80000L,
        80000L,
        24000L, 5600L,
        24000L, 4500L,
        4500L,
        24000L,
        10000L
      ),
      p = c(
        160L, 1800L,
        120L,
        120L,
        160L, 1500L,
        100L,
        100L,
        180L,
        650L, 1900L,
        450L, 2000L,
        2000L,
        450L,
        40L
      ),
      seed = c(
        601L, 602L, 610L, 611L, 603L, 604L, 612L, 613L, 614L,
        605L, 606L, 607L, 608L, 615L, 616L, 609L
      )
    )
  } else {
    data.frame(
      method = c(
        "mlxs_lm", "mlxs_lm",
        "mlxs_lm_summary",
        "mlxs_lm_predict",
        "mlxs_glm",
        "mlxs_glm_summary",
        "mlxs_glm_predict",
        "mlxs_glmnet_predict",
        "mlxs_cv_glmnet", "mlxs_prcomp",
        "mlxs_prcomp_summary",
        "mlxs_prcomp_predict",
        "mlxs_lm_bootstrap_summary"
      ),
      scenario = c(
        "lm_large_n", "lm_large_p",
        "lm_summary_large_n",
        "lm_predict_large_n",
        "glm_large_n",
        "glm_summary_large_n",
        "glm_predict_large_n",
        "glmnet_predict_large_n",
        "cv_glmnet_large_p", "prcomp_large_p",
        "prcomp_summary_large_p",
        "prcomp_predict_large_n",
        "lm_bootstrap_summary"
      ),
      n = c(
        100000L, 6500L, 65000L, 65000L, 65000L, 45000L, 45000L,
        50000L, 4500L, 3200L, 3200L, 18000L, 5000L
      ),
      p = c(
        110L, 1400L, 90L, 90L, 110L, 80L, 80L, 140L,
        1400L, 1600L, 1600L, 350L, 30L
      ),
      seed = c(
        501L, 502L, 510L, 511L, 503L, 512L, 513L,
        514L, 504L, 505L, 515L, 516L, 506L
      )
    )
  }
}

benchmark_regression_data <- function(seed, n, p, family = "gaussian") {
  x <- make_design(seed = seed, n = n, p = p, rho = 0.3)
  colnames(x) <- paste0("x", seq_len(p))
  beta <- numeric(p)
  active <- seq_len(min(12L, p))
  beta[active] <- seq(0.7, by = -0.05, length.out = length(active))
  eta <- drop(x %*% beta)
  if (identical(family, "binomial")) {
    prob <- plogis(eta / 8)
    y <- rbinom(n, size = 1L, prob = prob)
  } else {
    y <- 0.5 + eta + rnorm(n, sd = 5)
  }
  data <- data.frame(y = y, x, check.names = FALSE)
  formula <- reformulate(colnames(x), response = "y")
  list(data = data, formula = formula)
}

benchmark_glmnet_data <- function(seed, n, p) {
  x <- make_design(seed = seed, n = n, p = p, rho = 0.6)
  beta <- numeric(p)
  active <- seq(10L, p, by = 10L)
  beta[active] <- rnorm(length(active), sd = 0.35)
  y <- drop(x %*% beta + rnorm(n, sd = 5))
  colnames(x) <- paste0("x", seq_len(p))
  list(x = x, y = y)
}

benchmark_prcomp_data <- function(seed, n, p) {
  rank_true <- min(12L, max(4L, floor(min(n, p) / 4L)))
  prcomp_fuzz_case(
    seed = seed,
    scenario = "spiked",
    n = n,
    p = p,
    rank_true = rank_true,
    noise_sd = 0.01
  )
}

force_benchmark_fit <- function(fit, method) {
  switch(
    method,
    mlxs_lm = {
      Rmlx::mlx_eval(fit$coefficients)
    },
    mlxs_lm_summary = {
      Rmlx::mlx_eval(fit$std.error)
      Rmlx::mlx_eval(fit$statistic)
    },
    mlxs_lm_predict = {
      Rmlx::mlx_eval(fit)
    },
    mlxs_glm = {
      Rmlx::mlx_eval(fit$coefficients)
      stopifnot(fit$converged)
    },
    mlxs_glm_summary = {
      Rmlx::mlx_eval(fit$std.error)
      Rmlx::mlx_eval(fit$statistic)
    },
    mlxs_glm_predict = {
      Rmlx::mlx_eval(fit)
    },
    mlxs_glmnet_predict = {
      as.matrix(fit)
    },
    mlxs_cv_glmnet = {
      coef(fit, s = "lambda.min")
    },
    mlxs_prcomp = {
      Rmlx::mlx_eval(fit$sdev)
      Rmlx::mlx_eval(fit$rotation)
    },
    mlxs_prcomp_summary = {
      stopifnot(!is.null(fit$importance))
    },
    mlxs_prcomp_predict = {
      Rmlx::mlx_eval(fit)
    },
    mlxs_lm_bootstrap_summary = {
      Rmlx::mlx_eval(fit$std.error)
      stopifnot(!is.null(fit$bootstrap))
    },
    stop("Unknown benchmark method: ", method, call. = FALSE)
  )
  invisible(fit)
}

run_benchmark_case <- function(spec) {
  method <- spec$method
  n <- spec$n
  p <- spec$p
  seed <- spec$seed
  target <- switch(
    method,
    mlxs_lm = "fit",
    mlxs_glm = "fit",
    mlxs_cv_glmnet = "fit",
    mlxs_prcomp = "fit",
    mlxs_lm_summary = "summary",
    mlxs_glm_summary = "summary",
    mlxs_prcomp_summary = "summary",
    mlxs_lm_bootstrap_summary = "summary",
    mlxs_lm_predict = "prediction",
    mlxs_glm_predict = "prediction",
    mlxs_glmnet_predict = "prediction",
    mlxs_prcomp_predict = "prediction",
    stop("Unknown benchmark method: ", method, call. = FALSE)
  )
  fit_expr <- switch(
    method,
    mlxs_lm = {
      case <- benchmark_regression_data(seed, n, p)
      quote(mlxs_lm(case$formula, data = case$data))
    },
    mlxs_lm_summary = {
      case <- benchmark_regression_data(seed, n, p)
      fit <- mlxs_lm(case$formula, data = case$data)
      quote(summary(fit))
    },
    mlxs_lm_predict = {
      case <- benchmark_regression_data(seed, n, p)
      fit <- mlxs_lm(case$formula, data = case$data)
      quote(predict(fit, newdata = case$data))
    },
    mlxs_glm = {
      case <- benchmark_regression_data(seed, n, p, family = "binomial")
      quote(mlxs_glm(
        case$formula,
        data = case$data,
        family = mlxs_binomial(),
        control = list(maxit = 50, epsilon = 1e-5)
      ))
    },
    mlxs_glm_summary = {
      case <- benchmark_regression_data(seed, n, p, family = "binomial")
      fit <- mlxs_glm(
        case$formula,
        data = case$data,
        family = mlxs_binomial(),
        control = list(maxit = 50, epsilon = 1e-5)
      )
      quote(summary(fit))
    },
    mlxs_glm_predict = {
      case <- benchmark_regression_data(seed, n, p, family = "binomial")
      fit <- mlxs_glm(
        case$formula,
        data = case$data,
        family = mlxs_binomial(),
        control = list(maxit = 50, epsilon = 1e-5)
      )
      quote(predict(fit, newdata = case$data, type = "response"))
    },
    mlxs_glmnet_predict = {
      case <- benchmark_glmnet_data(seed, n, p)
      lambda <- exp(seq(log(0.3), log(0.003), length.out = 6L))
      fit <- mlxs_glmnet(
        case$x,
        case$y,
        family = mlxs_gaussian(),
        alpha = 0.5,
        lambda = lambda,
        standardize = FALSE,
        maxit = 1500L,
        tol = 1e-6
      )
      quote(predict(fit, newx = case$x, type = "response"))
    },
    mlxs_cv_glmnet = {
      case <- benchmark_glmnet_data(seed, n, p)
      lambda <- exp(seq(log(0.3), log(0.003), length.out = 6L))
      quote(mlxs_cv_glmnet(
        case$x,
        case$y,
        family = mlxs_gaussian(),
        alpha = 0.5,
        lambda = lambda,
        nfolds = 3L,
        standardize = FALSE,
        maxit = 1500L,
        tol = 1e-6
      ))
    },
    mlxs_prcomp = {
      x <- benchmark_prcomp_data(seed, n, p)
      quote(mlxs_prcomp(
        x,
        center = TRUE,
        scale. = FALSE,
        rank. = 12L,
        oversample = 10L,
        n_iter = 2L,
        seed = 1L
      ))
    },
    mlxs_prcomp_summary = {
      x <- benchmark_prcomp_data(seed, n, p)
      fit <- mlxs_prcomp(
        x,
        center = TRUE,
        scale. = FALSE,
        rank. = 12L,
        oversample = 10L,
        n_iter = 2L,
        seed = 1L
      )
      quote(summary(fit))
    },
    mlxs_prcomp_predict = {
      x <- benchmark_prcomp_data(seed, n, p)
      fit <- mlxs_prcomp(
        x,
        center = TRUE,
        scale. = FALSE,
        rank. = 12L,
        oversample = 10L,
        n_iter = 2L,
        seed = 1L
      )
      quote(predict(fit, newdata = x))
    },
    mlxs_lm_bootstrap_summary = {
      case <- benchmark_regression_data(seed, n, p)
      fit <- mlxs_lm(case$formula, data = case$data)
      bootstrap_B <- if (identical(fuzz_tier, "full")) 100L else 50L
      quote(summary(
        fit,
        bootstrap = TRUE,
        bootstrap_args = list(
          B = bootstrap_B,
          seed = seed,
          progress = FALSE
        )
      ))
    },
    stop("Unknown benchmark method: ", method, call. = FALSE)
  )

  elapsed <- system.time({
    fit <- eval(fit_expr)
    force_benchmark_fit(fit, method)
  })[["elapsed"]]
  bootstrap_B <- if (identical(method, "mlxs_lm_bootstrap_summary")) {
    fit$bootstrap$B
  } else {
    NA_integer_
  }

  fuzz_metric_rows(
    list(
      case_type = "benchmark",
      scenario = spec$scenario,
      n = n,
      p = p,
      method = method,
      bootstrap_B = bootstrap_B
    ),
    measure = "time",
    target = target,
    source = "mlx",
    aggregation = "elapsed_seconds",
    value = elapsed
  )
}

test_that("RmlxStats benchmark fuzz cases record elapsed time", {
  specs <- benchmark_specs(fuzz_tier)
  summaries <- vector("list", nrow(specs))
  for (idx in seq_len(nrow(specs))) {
    summaries[[idx]] <- run_benchmark_case(specs[idx, ])
  }
  summaries_df <- do.call(rbind, summaries)
  write_fuzz_summaries(
    summaries_df,
    suite = "mlxs-benchmarks",
    tier = fuzz_tier
  )

  expect_equal(nrow(summaries_df), nrow(specs))
  expect_true(all(summaries_df$measure == "time"))
  expect_true(all(summaries_df$target %in% c("fit", "summary", "prediction")))
  expect_true(all(summaries_df$aggregation == "elapsed_seconds"))
  expect_true(all(is.finite(summaries_df$value)))
  expect_true(all(summaries_df$value >= 0))
})
