#' Extract named coefficients from an MLX-backed model fit.
#'
#' @param fit A fitted model with a `coef()` method returning an MLX array.
#'
#' @return A named numeric vector.
#' @noRd
coef_vector <- function(fit) {
  coefs <- coef(fit)
  setNames(drop(as.matrix(coefs)), attr(coefs, "coef_names"))
}

#' Skip a fuzz test file unless the fuzz tier is enabled.
#'
#' @param subject Human-readable subject included in the skip message.
#'
#' @return The requested fuzz tier.
#' @noRd
skip_fuzz_tests <- function(subject) {
  fuzz_tier <- Sys.getenv("RMLXSTATS_RUN_FUZZ", unset = "")
  testthat::skip_if_not(
    fuzz_tier %in% c("fast", "full"),
    paste(
      "Set RMLXSTATS_RUN_FUZZ to 'fast' or 'full' to run",
      paste0(subject, " fuzz tests.")
    )
  )
  fuzz_tier
}

#' Run Monte Carlo replications with reproducible per-rep seeds.
#'
#' @param reps Number of replications.
#' @param seed0 Seed used to generate per-replication seeds.
#' @param rep_fun Function called once per replication. It must accept a
#'   `seed` argument.
#' @param label Short label used in error messages.
#' @param ... Extra arguments passed to `rep_fun`.
#'
#' @return A list containing one result per replication.
#' @noRd
run_mc_reps <- function(
  reps,
  seed0,
  rep_fun,
  label,
  ...
) {
  rep_fun_name <- substitute(rep_fun)
  if (!is.function(rep_fun)) {
    stop("`rep_fun` must be a function.", call. = FALSE)
  }
  set.seed(seed0)
  rep_seeds <- sample.int(.Machine$integer.max, reps)
  args <- list(...)
  results <- vector("list", reps)
  eval_env <- parent.frame()
  for (rep_idx in seq_len(reps)) {
    seed <- rep_seeds[[rep_idx]]
    rep_call <- as.call(c(list(rep_fun_name), list(seed = seed), args))
    results[[rep_idx]] <- tryCatch(
      eval(rep_call, envir = eval_env),
      error = function(err) {
        call_text <- deparse1(rep_call)
        stop(
          label,
          " failed for rep=",
          rep_idx,
          ", seed=",
          seed,
          ". Reproduce with ",
          call_text,
          ": ",
          conditionMessage(err),
          call. = FALSE
        )
      }
    )
  }
  results
}

#' Extract one vector-valued field from Monte Carlo results as a matrix.
#'
#' @param results List of per-replication results.
#' @param field Field name to extract from each result.
#' @param col_names Column names for the returned matrix.
#'
#' @return A matrix with one row per replication.
#' @noRd
mc_field_matrix <- function(results, field, col_names) {
  out <- do.call(rbind, lapply(results, `[[`, field))
  out <- as.matrix(out)
  colnames(out) <- col_names
  out
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

fuzz_metadata_columns <- function() {
  c(
    # Row family, e.g. deterministic, Monte Carlo, NIST, near-rank:
    "case_type",
    # Human-readable case name within the row family:
    "scenario",
    # Model family for GLM-style tests:
    "family",
    # Problem size:
    "n", "p",
    # Number of Monte Carlo replications:
    "nreps",
    # Elastic-net path metadata:
    "alpha", "lambda_index", "lambda",
    # PCA rank metadata:
    "rank", "rank_true",
    # Algorithm path, e.g. exact or randomized PCA:
    "method",
    # Synthetic noise and bootstrap settings:
    "noise_sd",
    "bootstrap_B"
  )
}

# Long-format fuzz result schema
#
# Each row records one metric for one fuzz case. The metadata columns
# (`case_type`, `scenario`, `family`, `n`, `p`, etc.) say which case produced
# the row. The metric columns say what `value` means:
#
# * `term`: optional coefficient/component name. `NA` means the value is a
#   case-level metric rather than a coefficient/component-level metric.
# * `measure`: broad statistical quantity.
#     - `truth`: true data-generating value.
#     - `estimate`: fitted estimate, usually averaged over Monte Carlo reps.
#     - `bias`: estimate minus truth.
#     - `error`: absolute, maximum, or RMSE discrepancy from `baseline`.
#     - `ratio`: relative discrepancy, such as relative PCA sdev RMSE.
#     - `standard_error`: coefficient SE from empirical/model/bootstrap source.
#     - `standard_error_ratio`: SE source divided by empirical SE.
#     - `coverage`: empirical CI coverage.
#     - `loss`: predictive or reconstruction loss.
#     - `delta`: signed difference from `baseline`; positive is worse when the
#       target is a loss/risk/objective.
#     - `selection`: active-set or support-recovery quantity.
#     - `diagnostic`: numerical or algorithm diagnostic, not a statistical
#       estimand.
# * `target`: object the metric is about.
#     - Regression: `coefficient`, `fitted`, `linear_predictor`, `vcov`,
#       `standard_error`, `residual_sigma`, `r_squared`, `deviance`, `aic`,
#       `confidence_interval`, `condition_number`, `convergence`, `iterations`,
#       `finite`, `bootstrap_failure`.
#     - Penalized regression: `prediction`, `risk`, `loss`, `objective`,
#       `active_set`, `active_size`, `true_positives`, `false_positives`,
#       `false_negatives`, `support_precision`, `support_recall`.
#     - PCA: `pca_sdev`, `subspace`, `reconstruction`, `explained_variance`,
#       `orthogonality`, `monotonicity`, `reproducibility`.
# * `source`: where the reported value came from.
#     - `mlx`: RmlxStats fit/result.
#     - `reference`: comparator implementation, usually stats or glmnet.
#     - `truth`: data-generating truth.
#     - `empirical`: Monte Carlo sampling distribution across fitted reps.
#     - `model`: model-reported analytic standard error.
#     - `bootstrap`: bootstrap estimate/diagnostic.
#     - `oracle`: data-generating prediction/loss.
#     - `design`: input design matrix diagnostic.
# * `baseline`: comparator named by the row.
#     - `truth`: compare with data-generating truth.
#     - `reference`: compare with stats/glmnet/prcomp reference result.
#     - `empirical`: compare with Monte Carlo empirical SE.
#     - `oracle`: compare with data-generating prediction/loss.
#     - `ideal`: compare with a mathematical ideal, e.g. orthogonality.
#     - `same_seed`: compare with the same randomized algorithm rerun.
#     - `NA`: no comparator; `value` is directly reported.
# * `aggregation`: how lower-level values were reduced into `value`.
#     - `value`: directly reported scalar.
#     - `mean`: average over replications, coefficients, or cases as implied by
#       the metadata/target.
#     - `max`: maximum over the relevant lower-level values.
#     - `rmse`: root mean squared error.
#     - `ratio`: numerator divided by denominator named by `baseline`.
#     - `delta`: source value minus baseline value.
#     - `count`: integer count.
#     - `rate`: fraction/proportion.
#     - `all`: logical all() result encoded as 1/0.
# * `value_se`: Monte Carlo standard error of `value`, when available; otherwise
#   `NA`.
#
# To interpret a row, read it as:
#   "For this metadata case and term, `value` is the `aggregation` of
#    `source`'s `measure` for `target`, compared with `baseline` if present."
fuzz_metric_columns <- function() {
  c(
    # Optional coefficient/component name:
    "term",
    # What kind of quantity was recorded, e.g. error, bias, coverage:
    "measure",
    # What the quantity applies to, e.g. coefficient, prediction, subspace:
    "target",
    # Which estimate produced the value, e.g. mlx, reference, empirical:
    "source",
    # Comparator for the value, e.g. truth, reference, empirical:
    "baseline",
    # How repeated values were reduced, e.g. max, mean, rmse:
    "aggregation",
    # Estimate and optional Monte Carlo standard error:
    "value", "value_se"
  )
}

fuzz_long_columns <- function() {
  c(
    "branch", "commit_hash", "datetime_utc", "tier", "suite",
    fuzz_metadata_columns(), fuzz_metric_columns()
  )
}

#' Build canonical long-format fuzz metric rows.
#'
#' @param metadata Named list or one-row data frame of case descriptors.
#' @param ... Metric columns. At minimum supply `measure`, `target`,
#'   `aggregation`, and `value`.
#'
#' @return A data frame with metadata plus metric columns.
#' @noRd
fuzz_metric_rows <- function(metadata, ...) {
  metadata <- as.data.frame(metadata, stringsAsFactors = FALSE)
  if (nrow(metadata) != 1L) {
    stop("`metadata` must describe exactly one fuzz case.", call. = FALSE)
  }
  metric_df <- data.frame(..., stringsAsFactors = FALSE)
  for (col in c("measure", "target", "aggregation", "value")) {
    if (!col %in% names(metric_df)) {
      stop("Missing fuzz metric column `", col, "`.", call. = FALSE)
    }
  }
  defaults <- list(
    term = NA_character_,
    source = "mlx",
    baseline = NA_character_,
    value_se = NA_real_
  )
  for (col in names(defaults)) {
    if (!col %in% names(metric_df)) {
      metric_df[[col]] <- defaults[[col]]
    }
  }
  metadata_cols <- fuzz_metadata_columns()
  for (col in setdiff(metadata_cols, names(metadata))) {
    metadata[[col]] <- NA
  }
  metadata <- metadata[metadata_cols]
  cbind(
    metadata[rep(1L, nrow(metric_df)), , drop = FALSE],
    metric_df[fuzz_metric_columns()],
    row.names = NULL
  )
}

write_fuzz_summaries <- function(summaries_df, suite, tier) {
  out_dir <- Sys.getenv("RMLXSTATS_FUZZ_OUT", unset = "")
  if (!nzchar(out_dir)) {
    message("Set RMLXSTATS_FUZZ_OUT to write fuzz summaries.")
    return(invisible(FALSE))
  }
  out_dir <- testthat::test_path(out_dir)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  path <- file.path(out_dir, "fuzz-results.csv")
  allowed <- setdiff(fuzz_long_columns(), c(
    "branch", "commit_hash", "datetime_utc", "tier", "suite"
  ))
  unknown <- setdiff(names(summaries_df), allowed)
  if (length(unknown)) {
    stop(
      "Unknown fuzz summary columns: ",
      paste(unknown, collapse = ", "),
      call. = FALSE
    )
  }
  for (col in setdiff(allowed, names(summaries_df))) {
    summaries_df[[col]] <- NA
  }
  summaries_df <- summaries_df[allowed]
  run_info <- data.frame(
    branch = git_value(
      c("rev-parse", "--abbrev-ref", "HEAD"),
      envvar = "GITHUB_REF_NAME"
    ),
    commit_hash = git_value(c("rev-parse", "HEAD"), envvar = "GITHUB_SHA"),
    datetime_utc = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
    tier = tier,
    suite = suite,
    stringsAsFactors = FALSE
  )
  summaries_df <- cbind(run_info, summaries_df)
  summaries_df <- summaries_df[fuzz_long_columns()]
  if (file.exists(path)) {
    existing_df <- utils::read.csv(
      path,
      check.names = FALSE,
      stringsAsFactors = FALSE
    )
    summaries_df <- rbind(existing_df[fuzz_long_columns()], summaries_df)
  }
  utils::write.table(
    summaries_df,
    file = path,
    sep = ",",
    row.names = FALSE,
    col.names = TRUE,
    append = FALSE,
    qmethod = "double"
  )
  message("Wrote fuzz summary to ", path)
  invisible(path)
}

glm_family_pair <- function(family) {
  switch(
    family,
    gaussian = list(base = gaussian(), mlx = mlxs_gaussian()),
    binomial = list(base = binomial(), mlx = mlxs_binomial()),
    poisson = list(base = poisson(), mlx = mlxs_poisson())
  )
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
  # Generates the covariance/correlation matrix. Diagonals are 1.
  sigma <- outer(idx, idx, function(i, j) rho ^ abs(i - j))
  # C = chol(sigma) satisfies t(C) %*% C = sigma, so covariance
  # of the below is cov(M) = t(C) %*% I %*% C = sigma.
  matrix(rnorm(n * p), nrow = n) %*% chol(sigma)
}

#' Generate a full-rank synthetic regression test case.
#'
#' @param seed Integer seed used to make the case reproducible.
#' @param n Number of observations.
#' @param p Number of predictors.
#' @param rho AR(1) correlation parameter passed to `make_design()`.
#' @param noise Standard deviation of Gaussian response noise.
#' @param family Response family to simulate.
#' @param intercept Intercept used in the linear predictor.
#' @param beta Optional coefficient vector. Defaults to a simple decreasing
#'   sequence.
#' @param poisson_overdispersion Gamma mixture variance multiplier for Poisson
#'   responses. Use `1` for ordinary Poisson data.
#'
#' @return A list with `data`, `formula`, and `truth` entries.
#' @noRd
make_case <- function(
  seed,
  n = 64,
  p = 4,
  rho = 0,
  noise = 0.4,
  family = c("gaussian", "binomial", "poisson"),
  intercept = 1,
  beta = NULL,
  poisson_overdispersion = 1
) {
  family <- match.arg(family)
  set.seed(seed)
  x <- make_design(n = n, p = p, rho = rho)
  ind_vars <- paste0("x", seq_len(p))
  colnames(x) <- ind_vars
  if (is.null(beta)) {
    beta <- seq(0.8, by = -0.25, length.out = p)
  }
  eta <- intercept + drop(x %*% beta)

  # Keep simulated non-Gaussian means moderate unless a test deliberately
  # overrides the intercept or coefficients.
  y <- switch(
    family,
    gaussian = eta + rnorm(n, sd = noise),
    binomial = rbinom(n, size = 1, prob = plogis(eta)),
    poisson = {
      lambda <- exp(pmin(eta, 2.2))
      if (poisson_overdispersion > 1) {
        shape <- 1 / (poisson_overdispersion - 1)
        lambda <- lambda * rgamma(n, shape = shape, rate = shape)
      }
      rpois(n, lambda = lambda)
    }
  )

  data <- data.frame(y = y, x)
  list(
    data = data,
    formula = reformulate(ind_vars, response = "y"),
    truth = setNames(c(intercept, beta), c("(Intercept)", ind_vars))
  )
}
