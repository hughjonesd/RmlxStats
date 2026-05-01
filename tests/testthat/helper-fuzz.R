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

write_fuzz_summaries <- function(summaries_df, suite, tier) {
  out_dir <- Sys.getenv("RMLXSTATS_FUZZ_OUT", unset = "")
  out_dir <- testthat::test_path(out_dir)
  if (!nzchar(out_dir)) {
    message("Set RMLXSTATS_FUZZ_OUT to write fuzz summaries.")
    return(invisible(FALSE))
  }
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  path <- file.path(out_dir, paste0(suite, "-", tier, ".csv"))
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
  possible_fuzz_columns <- c(
    # Summarised row family, e.g. large design, NIST, near-rank, Monte Carlo:
    "case_type",
    # Human-readable case name within the row family:
    "scenario",
    # GLM response family, e.g. gaussian, binomial, poisson:
    "family",
    # Number of observations in the fitted model:
    "n",
    # Number of coefficients in the fitted model matrix:
    "p",
    # Number of Monte Carlo replications:
    "nreps",
    # Elastic-net mixing parameter:
    "alpha",
    # Lambda position in the fitted path:
    "lambda_index",
    # Penalization strength:
    "lambda",
    # Number of principal components retained:
    "rank",
    # True generating rank for PCA-style synthetic data:
    "rank_true",
    # PCA algorithm path, e.g. exact or randomized:
    "method",
    # Noise standard deviation used by synthetic generators:
    "noise_sd",
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
    # Mean fraction of Monte Carlo fits that converged:
    "convergence_rate",
    # Mean number of iterations over Monte Carlo replications:
    "mean_iterations",
    # Maximum number of iterations over Monte Carlo replications:
    "max_iterations",
    # Base-R condition number of the model matrix. A matrix with a big
    # condition number is nearly linearly dependent.
    "condition_number",
    # Largest absolute coefficient difference from the reference:
    "max_coef_error",
    # Largest absolute fitted-value difference from the reference:
    "max_fitted_error",
    # Largest absolute linear-predictor difference from the reference:
    "max_eta_error",
    # Largest absolute variance-covariance difference from the reference:
    "max_vcov_error",
    # Absolute deviance difference from the reference:
    "deviance_error",
    # Absolute AIC difference from the reference:
    "aic_error",
    # Test-set prediction loss for MLX-backed penalized fits:
    "test_loss",
    # Test-set prediction loss for the reference implementation:
    "reference_test_loss",
    # Test-set prediction loss under the true data-generating model:
    "oracle_test_loss",
    # Test loss minus oracle test loss:
    "excess_risk",
    # Absolute test-loss difference from the reference:
    "loss_error",
    # Relative test-loss difference from the reference:
    "relative_loss_error",
    # Number of nonzero coefficients in a penalized fit:
    "active_size",
    # Number of selected coefficients with nonzero true coefficient:
    "true_positives",
    # Number of selected coefficients with zero true coefficient:
    "false_positives",
    # Number of nonzero true coefficients missed by the fit:
    "false_negatives",
    # true_positives / active_size, with NA when undefined:
    "support_precision",
    # true_positives / number of true active coefficients, with NA when
    # undefined:
    "support_recall",
    # Largest absolute prediction difference from the reference:
    "max_prediction_error",
    # Largest absolute penalized objective difference from the reference:
    "max_objective_error",
    # Largest absolute PCA standard-deviation difference from the reference:
    "max_sdev_error",
    # Relative RMSE of PCA standard deviations against the reference:
    "relative_sdev_rmse",
    # Relative Frobenius distance between estimated and reference projection
    # matrices VV'. This is sign-invariant and, unlike raw loadings, is stable
    # when PCA columns can flip signs:
    "subspace_error",
    # Relative centred/scaled reconstruction error, ||X - scores V'|| / ||X||,
    # for the MLX fit:
    "reconstruction_error",
    # Relative centred/scaled reconstruction error for the reference fit:
    "reference_reconstruction_error",
    # Reconstruction error minus reference reconstruction error:
    "excess_reconstruction_error",
    # Largest absolute entry of V'V - I, measuring whether loadings are
    # orthonormal:
    "orthogonality_error",
    # Largest explained-variance ratio difference from the reference:
    "explained_variance_error",
    # Largest increase in reconstruction error over a rank ladder:
    "monotonicity_error",
    # Largest same-seed randomized PCA difference across the checked outputs:
    # standard deviations and the loading subspace projector:
    "reproducibility_error",
    # Largest absolute standard-error difference from certified values:
    "max_se_error",
    # Absolute residual standard error difference from certified value.
    # sigma is the estimated s.e. of residuals,
    # sigma = sqrt(sum(residuals^2) / df.residual)
    # where df.residual = n-p:
    "sigma_error",
    # Absolute R-squared difference from certified value:
    "r_squared_error",
    # Whether the MLX-backed model reported convergence:
    "converged",
    # Number of fitting iterations used:
    "iterations",
    # Whether all checked MLX-backed numeric outputs are finite:
    "all_finite"
  )
  unknown_columns <- setdiff(names(summaries_df), possible_fuzz_columns)
  if (length(unknown_columns)) {
    stop(
      "Unknown fuzz summary columns: ",
      paste(unknown_columns, collapse = ", "),
      call. = FALSE
    )
  }
  summary_columns <- intersect(possible_fuzz_columns, names(summaries_df))
  summaries_df <- summaries_df[summary_columns]
  summaries_df <- cbind(run_info, summaries_df)
  file_exists <- file.exists(path)
  if (file_exists) {
    existing_df <- utils::read.csv(
      path,
      check.names = FALSE,
      stringsAsFactors = FALSE
    )
    # Different blocks in a test file may record different valid metrics.
    # Widen the suite file when needed, so the CSV never has ragged rows.
    output_columns <- union(names(existing_df), names(summaries_df))
    for (col in setdiff(output_columns, names(existing_df))) {
      existing_df[[col]] <- NA
    }
    for (col in setdiff(output_columns, names(summaries_df))) {
      summaries_df[[col]] <- NA
    }
    summaries_df <- rbind(
      existing_df[output_columns],
      summaries_df[output_columns]
    )
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
