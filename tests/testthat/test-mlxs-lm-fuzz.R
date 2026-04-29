fuzz_tier <- Sys.getenv("RMLXSTATS_RUN_FUZZ", unset = "")
run_lm_fuzz <- fuzz_tier %in% c("fast", "full")

testthat::skip_if_not(
  run_lm_fuzz,
  paste(
    "Set RMLXSTATS_RUN_FUZZ to 'fast' or 'full' to run",
    "mlxs_lm fuzz tests."
  )
)

coef_vector <- function(fit) {
  coefs <- coef(fit)
  setNames(drop(as.matrix(coefs)), attr(coefs, "coef_names"))
}

expect_mlxs_lm_matches_lm <- function(
  formula,
  data,
  weights = NULL,
  coef_tol = 1e-4,
  value_tol = 1e-5,
  label = ""
) {
  base_fit <- if (is.null(weights)) {
    lm(formula, data = data)
  } else {
    data$.weights <- weights
    lm(formula, data = data, weights = .weights)
  }
  mlx_fit <- if (is.null(weights)) {
    mlxs_lm(formula, data = data)
  } else {
    data$.weights <- weights
    mlxs_lm(formula, data = data, weights = .weights)
  }

  expect_equal(coef_vector(mlx_fit), coef(base_fit),
               tolerance = coef_tol, ignore_attr = TRUE, info = label)
  expect_equal(drop(as.matrix(fitted(mlx_fit))), fitted(base_fit),
               tolerance = value_tol, ignore_attr = TRUE, info = label)
  expect_equal(drop(as.matrix(residuals(mlx_fit))), residuals(base_fit),
               tolerance = value_tol, ignore_attr = TRUE, info = label)
  expect_equal(as.matrix(vcov(mlx_fit)), vcov(base_fit),
               tolerance = coef_tol, ignore_attr = TRUE, info = label)
  expect_equal(confint(mlx_fit), confint(base_fit),
               tolerance = coef_tol, ignore_attr = TRUE, info = label)
  expect_equal(mlx_fit$rank, base_fit$rank, info = label)
  expect_equal(mlx_fit$df.residual, base_fit$df.residual, info = label)

  list(base = base_fit, mlx = mlx_fit)
}

#' Generate a regression design matrix for fuzz tests.
#'
#' @param seed Optional integer seed. When `NULL`, use the current RNG stream.
#' @param n Number of observations.
#' @param p Number of predictors.
#' @param rho AR(1) correlation parameter. Use `0` for independent predictors.
#'
#' @return An `n` by `p` numeric matrix.
#' @noRd
make_design <- function(seed = NULL, n, p, rho = 0) {
  if (!is.null(seed)) {
    set.seed(seed)
  }
  if (rho == 0) {
    return(matrix(rnorm(n * p), nrow = n))
  }
  idx <- seq_len(p)
  # generates the covariance/correlation matrix. Diagonals are 1:
  sigma <- outer(idx, idx, function(i, j) rho ^ abs(i - j))
  # C = chol(sigma) satisfies t(C) %*% C = sigma.
  # so covariance of the below is cov(M) = t(C) %*% I %*% C = sigma:
  matrix(rnorm(n * p), nrow = n) %*% chol(sigma)
}

#' Generate a full-rank synthetic `mlxs_lm()` test case.
#'
#' @param seed Integer seed used to make the case reproducible.
#' @param n Number of observations.
#' @param p Number of predictors.
#' @param rho AR(1) correlation parameter passed to `make_design()`.
#' @param noise Standard deviation of the Gaussian response noise.
#'
#' @return A list with `data` and `formula` entries.
#' @noRd
make_case <- function(seed, n = 64, p = 4, rho = 0, noise = 0.4) {
  x <- make_design(seed, n, p, rho)
  beta <- seq(0.8, by = -0.25, length.out = p)
  y <- 1 + drop(x %*% beta) + rnorm(n, sd = noise)
  data <- data.frame(y = y, x)
  ind_vars <- paste0("x", seq_len(p))
  names(data) <- c("y", ind_vars)
  fml <- stats::reformulate(ind_vars, response = "y")
  list(data = data, formula = fml)
}

git_value <- function(args, envvar = NULL, fallback = NA_character_) {
  if (!is.null(envvar)) {
    env_value <- Sys.getenv(envvar, unset = "")
    if (nzchar(env_value)) {
      return(env_value)
    }
  }
  out <- tryCatch(
    system2("git", args = args, stdout = TRUE, stderr = FALSE),
    error = function(err) character()
  )
  if (length(out) && nzchar(out[[1]])) out[[1]] else fallback
}

write_fuzz_summary <- function(summary) {
  out_dir <- Sys.getenv("RMLXSTATS_FUZZ_OUT", unset = "")
  if (!nzchar(out_dir)) {
    message("Set RMLXSTATS_FUZZ_OUT to write fuzz summaries.")
    return(invisible(FALSE))
  }
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  path <- file.path(out_dir, paste0("mlxs-lm-", fuzz_tier, ".csv"))
  run_info <- data.frame(
    branch = git_value(
      c("rev-parse", "--abbrev-ref", "HEAD"),
      envvar = "GITHUB_REF_NAME"
    ),
    commit_hash = git_value(c("rev-parse", "HEAD"), envvar = "GITHUB_SHA"),
    datetime_utc = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
    tier = fuzz_tier,
    stringsAsFactors = FALSE
  )
  fuzz_summary_columns <- c(
    # Summary row family, e.g. large design, NIST, near-rank, Monte Carlo:
    "case_type",
    # Human-readable case name within the row family:
    "scenario",
    # Number of observations in the fitted model:
    "n",
    # Number of coefficients in the fitted model matrix:
    "p",
    # Number of Monte Carlo replications:
    "nreps",
    # Coefficient name for coefficient-level Monte Carlo rows:
    "coefficient",
    # True data-generating coefficient value for Monte Carlo rows:
    "truth",
    # Mean estimate over Monte Carlo replications:
    "mean_estimate",
    # Monte Carlo mean estimate minus truth:
    "bias",
    # Monte Carlo standard error of the bias estimate:
    "mcse_bias",
    # Monte Carlo root mean squared error against truth:
    "rmse",
    # Empirical standard deviation of estimates across replications.
    # This is (an estimate of) the true value of the standard error:
    "empirical_se",
    # Mean model-reported standard error across replications. In
    # homoskedastic cases, should be close to empirical_se:
    "average_model_se",
    # Empirical confidence-interval coverage across replications:
    "ci_coverage",
    # Monte Carlo standard error of the coverage estimate:
    "mcse_coverage",
    # Base-R condition number of the model matrix. A matrix with a big
    # condition number is nearly linearly dependent.
    "condition_number",
    # Largest absolute coefficient difference from the reference:
    "max_coef_error",
    # Largest absolute fitted-value difference from the reference:
    "max_fitted_error",
    # Largest absolute variance-covariance difference from the reference:
    "max_vcov_error",
    # Largest absolute standard-error difference from certified values:
    "max_se_error",
    # Absolute residual standard error difference from certified value.
    # sigma is the estimated s.e. of residuals,
    # sigma = sqrt(sum(residuals^2) / df.residual) 
    # where df.residual = n-p:
    "sigma_error",
    # Absolute R-squared difference from certified value:
    "r_squared_error",
    # Whether all checked MLX-backed numeric outputs are finite:
    "all_finite"
  )
  for (col in setdiff(fuzz_summary_columns, names(summary))) {
    summary[[col]] <- NA
  }
  summary <- summary[fuzz_summary_columns]
  summary <- cbind(run_info, summary)
  file_exists <- file.exists(path)
  utils::write.table(
    summary,
    file = path,
    sep = ",",
    row.names = FALSE,
    col.names = !file_exists,
    append = file_exists,
    qmethod = "double"
  )
  message("Wrote fuzz summary to ", path)
  invisible(path)
}

lm_difference_summary <- function(scenario, formula, data, case_type) {
  mlx_fit <- mlxs_lm(formula, data = data)
  base_fit <- lm(formula, data = data)
  mlx_coef <- coef_vector(mlx_fit)
  mlx_vcov <- as.matrix(vcov(mlx_fit))
  mlx_fitted <- drop(as.matrix(fitted(mlx_fit)))
  finite_values <- c(mlx_coef, mlx_vcov, mlx_fitted)

  data.frame(
    case_type = case_type,
    scenario = scenario,
    n = nrow(model.frame(formula, data)),
    p = ncol(model.matrix(formula, data)),
    condition_number = kappa(model.matrix(formula, data)),
    max_coef_error = max(abs(mlx_coef - coef(base_fit))),
    max_fitted_error = max(abs(mlx_fitted - fitted(base_fit))),
    max_vcov_error = max(abs(mlx_vcov - vcov(base_fit))),
    all_finite = all(is.finite(finite_values)),
    stringsAsFactors = FALSE
  )
}

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

test_that("mlxs_lm deterministic differential fuzz cases match stats::lm", {
  cases <- list(
    iid = make_case(seed = 1, n = 48, p = 4),
    ar1_high_rho = make_case(seed = 17, n = 72, p = 5, rho = 0.92),
    leverage = {
      obj <- make_case(seed = 2718, n = 64, p = 4)
      obj$data[1, c("x1", "x2")] <- c(8, -7)
      obj$data$y[1] <- obj$data$y[1] + 3
      obj
    },
    near_collinear = {
      set.seed(314159)
      n <- 80
      x1 <- rnorm(n)
      x2 <- x1 + rnorm(n, sd = 0.03)
      x3 <- rnorm(n)
      y <- 1 + 0.8 * x1 - 0.5 * x2 + 0.25 * x3 + rnorm(n, sd = 0.35)
      list(data = data.frame(y, x1, x2, x3), formula = y ~ x1 + x2 + x3)
    }
  )

  for (case_name in names(cases)) {
    case <- cases[[case_name]]
    expect_mlxs_lm_matches_lm(
      case$formula,
      case$data,
      coef_tol = 1e-3,
      value_tol = 1e-4,
      label = case_name
    )
  }

  weighted <- make_case(seed = 8675309, n = 60, p = 3)
  weights <- seq(0.25, 1.5, length.out = nrow(weighted$data))
  expect_mlxs_lm_matches_lm(
    weighted$formula,
    weighted$data,
    weights = weights,
    coef_tol = 1e-5,
    value_tol = 1e-5,
    label = "weighted"
  )

  specs <- data.frame(
    scenario = c("large_n", "large_p"),
    seed = c(101L, 102L),
    n = c(20000L, 10000L),
    p = c(30L, 300L)
  )
  if (identical(fuzz_tier, "full")) {
    specs <- rbind(
      specs,
      data.frame(scenario = "larger_p", seed = 103L, n = 30000L, p = 500L)
    )
  }

  summaries <- vector("list", nrow(specs))
  for (idx in seq_len(nrow(specs))) {
    spec <- specs[idx, ]
    set.seed(spec$seed)
    x <- make_design(n = spec$n, p = spec$p)
    beta <- numeric(spec$p)
    active <- seq_len(min(10L, spec$p))
    beta[active] <- seq(1, by = -0.12, length.out = length(active))
    y <- 0.5 + drop(x %*% beta) + rnorm(spec$n, sd = 0.25)
    data <- data.frame(y = y, x)
    names(data) <- c("y", paste0("x", seq_len(spec$p)))
    formula <- as.formula(
      paste("y ~", paste(names(data)[-1L], collapse = " + "))
    )

    expect_mlxs_lm_matches_lm(
      formula,
      data,
      coef_tol = 1e-5,
      value_tol = 1e-5,
      label = spec$scenario
    )
    summaries[[idx]] <- lm_difference_summary(
      scenario = spec$scenario,
      formula = formula,
      data = data,
      case_type = "large_design"
    )
  }

  summary <- do.call(rbind, summaries)
  print(summary, digits = 4)
  write_fuzz_summary(summary)
})

test_that("mlxs_lm metamorphic fuzz properties hold", {
  case <- make_case(seed = 17, n = 70, p = 3, rho = 0.5)
  data <- case$data
  fit <- mlxs_lm(case$formula, data = data)

  set.seed(2718)
  perm <- sample(seq_len(nrow(data)))
  perm_fit <- mlxs_lm(case$formula, data = data[perm, ])
  expect_equal(coef_vector(perm_fit), coef_vector(fit),
               tolerance = 1e-5, ignore_attr = TRUE)
  expect_equal(drop(as.matrix(fitted(perm_fit)))[order(perm)],
               drop(as.matrix(fitted(fit))),
               tolerance = 1e-5, ignore_attr = TRUE)

  col_fit <- mlxs_lm(y ~ x3 + x1 + x2, data = data)
  expect_equal(coef_vector(col_fit)[names(coef_vector(fit))],
               coef_vector(fit),
               tolerance = 1e-5, ignore_attr = TRUE)
  expect_equal(drop(as.matrix(predict(col_fit, newdata = data))),
               drop(as.matrix(predict(fit, newdata = data))),
               tolerance = 1e-5, ignore_attr = TRUE)

  expect_equal(drop(as.matrix(predict(fit))),
               drop(as.matrix(predict(fit, newdata = data))),
               tolerance = 1e-5, ignore_attr = TRUE)
})

test_that("mlxs_lm rank-deficient fuzz cases fail clearly", {
  data <- make_case(seed = 1, n = 40, p = 3)$data
  data$x_dup <- data$x1
  expect_error(
    mlxs_lm(y ~ x1 + x2 + x_dup, data = data),
    "full-rank model matrix",
    fixed = TRUE
  )

  data$x_combo <- data$x1 + data$x2
  expect_error(
    mlxs_lm(y ~ x1 + x2 + x3 + x_combo, data = data),
    "full-rank model matrix",
    fixed = TRUE
  )
})

test_that("mlxs_lm near-rank-deficient stability is tracked", {
  set.seed(1717)
  n <- 160
  x1 <- rnorm(n)
  x3 <- rnorm(n)
  z <- rnorm(n)
  noise <- rnorm(n, sd = 0.2)
  eps_grid <- if (identical(fuzz_tier, "full")) {
    c(1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
  } else {
    c(1e-1, 1e-2, 1e-3, 1e-4)
  }

  summaries <- vector("list", length(eps_grid))
  for (idx in seq_along(eps_grid)) {
    eps <- eps_grid[[idx]]
    x2 <- x1 + eps * z
    y <- 1 + 0.8 * x1 - 0.5 * x2 + 0.25 * x3 + noise
    data <- data.frame(y, x1, x2, x3)
    scenario <- paste0("epsilon_", format(eps, scientific = TRUE))
    summaries[[idx]] <- lm_difference_summary(
      scenario = scenario,
      formula = y ~ x1 + x2 + x3,
      data = data,
      case_type = "near_rank_deficient"
    )
  }

  summary <- do.call(rbind, summaries)
  print(summary, digits = 4)
  write_fuzz_summary(summary)

  expect_true(all(summary$all_finite), info = "non-finite near-singular fit")
  expect_true(
    all(summary$max_fitted_error <= 0.05),
    info = paste(summary$scenario[summary$max_fitted_error > 0.05],
                 collapse = ", ")
  )

  moderate <- summary$condition_number <= 500
  expect_true(
    all(summary$max_coef_error[moderate] <= 1e-3),
    info = paste(summary$scenario[
      moderate & summary$max_coef_error > 1e-3
    ], collapse = ", ")
  )
  expect_true(
    all(summary$max_vcov_error[moderate] <= 1e-3),
    info = paste(summary$scenario[
      moderate & summary$max_vcov_error > 1e-3
    ], collapse = ", ")
  )
})

test_that("mlxs_lm NIST StRD fixtures are checked", {
  # Norris is a lower-difficulty certified regression check where MLX should
  # match the published coefficients, SEs, sigma, and R-squared. Longley is a
  # classic multicollinearity stress case; Wampler1 is an exact polynomial fit
  # with severe scaling. Those harder cases are tracked as finite-output and
  # R-squared diagnostics rather than strict single-precision coefficient
  # or standard-error checks.
  source(
    testthat::test_path("fixtures", "nist-strd", "linear-regression.R"),
    local = TRUE
  )

  summaries <- vector("list", length(nist_lm_cases))
  names(summaries) <- names(nist_lm_cases)
  for (case_name in names(nist_lm_cases)) {
    case <- nist_lm_cases[[case_name]]
    fit <- mlxs_lm(case$formula, data = case$data)
    fit_summary <- summary(fit)
    coefs <- coef_vector(fit)
    se <- setNames(as.numeric(fit_summary$std.error), names(coefs))
    sigma <- fit_summary$sigma
    r_squared <- fit_summary$r.squared
    finite_values <- c(coefs, se, sigma, r_squared)

    summaries[[case_name]] <- data.frame(
      case_type = "nist_strd",
      scenario = case$name,
      n = nrow(model.frame(case$formula, case$data)),
      p = ncol(model.matrix(case$formula, case$data)),
      condition_number = kappa(model.matrix(case$formula, case$data)),
      max_coef_error = max(abs(coefs[names(case$coef)] - case$coef)),
      max_se_error = max(abs(se[names(case$se)] - case$se)),
      sigma_error = abs(sigma - case$sigma),
      r_squared_error = abs(r_squared - case$r_squared),
      all_finite = all(is.finite(finite_values)),
      stringsAsFactors = FALSE
    )

    expect_true(all(is.finite(finite_values)), info = case$name)
    if (isTRUE(case$certified)) {
      expect_equal(coefs[names(case$coef)], case$coef,
                   tolerance = 1e-4, ignore_attr = TRUE, info = case$name)
      expect_equal(se[names(case$se)], case$se,
                   tolerance = 1e-5, ignore_attr = TRUE, info = case$name)
      expect_equal(sigma, case$sigma, tolerance = 1e-5, info = case$name)
      expect_equal(r_squared, case$r_squared,
                   tolerance = 1e-8, info = case$name)
    } else {
      expect_equal(r_squared, case$r_squared,
                   tolerance = 1e-5, info = case$name)
    }
  }

  summary <- do.call(rbind, summaries)
  print(summary, digits = 4)
  write_fuzz_summary(summary)
})

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
  summary <- rbind(hom, het)

  print(summary, digits = 4)
  write_fuzz_summary(summary)

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

  coverage_band <- 4 * sqrt(0.95 * 0.05 / hom_reps)
  expect_true(
    all(abs(hom$ci_coverage - 0.95) <= coverage_band),
    info = paste(
      "homoskedastic coverage outside Monte Carlo band:",
      paste(hom$coefficient[abs(hom$ci_coverage - 0.95) > coverage_band],
            collapse = ", ")
    )
  )
})
