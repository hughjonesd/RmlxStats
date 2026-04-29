fuzz_tier <- Sys.getenv("RMLXSTATS_RUN_FUZZ", unset = "")
run_lm_fuzz <- fuzz_tier %in% c("lm-fast", "lm-long")

skip_lm_fuzz <- function() {
  testthat::skip_if_not(
    run_lm_fuzz,
    paste(
      "Set RMLXSTATS_RUN_FUZZ to 'lm-fast' or 'lm-long' to run",
      "mlxs_lm fuzz tests."
    )
  )
}

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

#' Generate a small regression design matrix for fuzz tests.
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
  sigma <- outer(idx, idx, function(i, j) rho ^ abs(i - j))
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
  names(data) <- c("y", paste0("x", seq_len(p)))
  list(data = data, formula = as.formula(
    paste("y ~", paste(names(data)[-1L], collapse = " + "))
  ))
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
  summary <- cbind(
    run_info[rep(1L, nrow(summary)), , drop = FALSE],
    summary
  )
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

summarise_lm_mc <- function(estimates, ses, covered, truth, scenario, reps) {
  out <- data.frame(
    scenario = scenario,
    coefficient = names(truth),
    truth = unname(truth),
    mean_estimate = colMeans(estimates),
    bias = colMeans(estimates) - unname(truth),
    rmse = sqrt(colMeans((sweep(estimates, 2, unname(truth)))^2)),
    empirical_se = apply(estimates, 2, stats::sd),
    average_model_se = colMeans(ses),
    ci_coverage = colMeans(covered),
    mcse_coverage = sqrt(0.95 * 0.05 / reps),
    row.names = NULL
  )
  out
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
    set.seed(rep_seeds[rep_idx])
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

    fit <- tryCatch(
      mlxs_lm(y ~ x1 + x2 + x3, data = data),
      error = function(err) {
        stop(
          "run_lm_mc failed for scenario='",
          scenario,
          "', rep=",
          rep_idx,
          ", seed=",
          rep_seeds[rep_idx],
          ": ",
          conditionMessage(err),
          call. = FALSE
        )
      }
    )
    ci <- confint(fit)
    sum_fit <- summary(fit)

    estimates[rep_idx, ] <- coef_vector(fit)
    ses[rep_idx, ] <- as.numeric(sum_fit$std.error)
    covered[rep_idx, ] <- ci[, 1] <= truth & truth <= ci[, 2]
  }

  stopifnot(!anyNA(covered))

  summarise_lm_mc(estimates, ses, covered, truth, scenario, reps)
}

test_that("mlxs_lm deterministic fuzz cases match stats::lm", {
  skip_lm_fuzz()

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
})

test_that("mlxs_lm metamorphic fuzz properties hold", {
  skip_lm_fuzz()

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
  skip_lm_fuzz()

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

test_that("mlxs_lm Monte Carlo fuzz summaries are within tolerance", {
  skip_lm_fuzz()

  reps <- if (identical(fuzz_tier, "lm-long")) 1000L else 200L
  hom <- run_lm_mc(reps = reps, seed0 = 10000, scenario = "homoskedastic")
  het <- run_lm_mc(
    reps = if (identical(fuzz_tier, "lm-long")) 500L else 100L,
    seed0 = 20000,
    scenario = "heteroskedastic"
  )
  summary <- rbind(hom, het)

  print(summary, digits = 4)
  write_fuzz_summary(summary)

  hom_bias_mcse <- hom$empirical_se / sqrt(reps)
  expect_true(
    all(abs(hom$bias) <= 4 * hom_bias_mcse + 0.01),
    info = paste(
      "homoskedastic bias outside Monte Carlo band:",
      paste(hom$coefficient[abs(hom$bias) > 4 * hom_bias_mcse + 0.01],
            collapse = ", ")
    )
  )

  coverage_band <- 4 * sqrt(0.95 * 0.05 / reps) + 0.01
  expect_true(
    all(abs(hom$ci_coverage - 0.95) <= coverage_band),
    info = paste(
      "homoskedastic coverage outside Monte Carlo band:",
      paste(hom$coefficient[abs(hom$ci_coverage - 0.95) > coverage_band],
            collapse = ", ")
    )
  )
})
