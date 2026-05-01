#' Compute prediction loss for glmnet fuzz tests.
#'
#' @param y Observed response.
#' @param pred Predicted mean response. For binomial models, this must be a
#'   predicted probability.
#' @param family Model family.
#'
#' @return Mean squared prediction error for Gaussian fits, or mean binomial
#'   negative log-likelihood for binomial fits.
#' @noRd
glmnet_fuzz_loss <- function(y, pred, family = c("gaussian", "binomial")) {
  family <- match.arg(family)
  pred <- as.numeric(pred)
  if (family == "gaussian") {
    return(mean((y - pred)^2))
  }
  prob <- pmin(pmax(pred, 1e-8), 1 - 1e-8)
  -mean(y * log(prob) + (1 - y) * log(1 - prob))
}

#' Compute the elastic-net training objective.
#'
#' @param x Training design matrix.
#' @param y Training response.
#' @param beta Coefficient vector, excluding intercept.
#' @param a0 Intercept.
#' @param lambda Penalization strength.
#' @param alpha Elastic-net mixing parameter.
#' @param family Model family.
#'
#' @return Average Gaussian squared-error loss divided by two, or average
#'   binomial negative log-likelihood, plus the elastic-net penalty.
#' @noRd
glmnet_fuzz_objective <- function(
  x,
  y,
  beta,
  a0,
  lambda,
  alpha,
  family = c("gaussian", "binomial")
) {
  family <- match.arg(family)
  eta <- drop(x %*% beta + a0)
  pred <- if (family == "gaussian") eta else 1 / (1 + exp(-eta))
  loss <- glmnet_fuzz_loss(y, pred, family = family)
  if (family == "gaussian") {
    # glmnet's Gaussian elastic-net objective uses RSS / (2n), so divide
    # mean squared error by two before adding the penalty.
    loss <- loss / 2
  }
  penalty <- lambda * (
    alpha * sum(abs(beta)) + (1 - alpha) * sum(beta^2) / 2
  )
  loss + penalty
}

#' Summarise support recovery for a penalized fit.
#'
#' @param beta Estimated coefficient vector, excluding intercept.
#' @param truth True coefficient vector.
#' @param threshold Absolute value above which coefficients count as active.
#'
#' @return A one-row data frame with active-set size, true positives, false
#'   positives, false negatives, precision, and recall.
#' @noRd
glmnet_fuzz_support <- function(beta, truth, threshold = 1e-7) {
  selected <- abs(beta) > threshold
  active <- abs(truth) > threshold
  true_positives <- sum(selected & active)
  false_positives <- sum(selected & !active)
  false_negatives <- sum(!selected & active)
  active_size <- sum(selected)
  n_active <- sum(active)
  data.frame(
    active_size = active_size,
    true_positives = true_positives,
    false_positives = false_positives,
    false_negatives = false_negatives,
    support_precision = if (active_size > 0) {
      true_positives / active_size
    } else {
      NA_real_
    },
    support_recall = if (n_active > 0) {
      true_positives / n_active
    } else {
      NA_real_
    }
  )
}

#' Generate sparse truth coefficients for glmnet fuzz tests.
#'
#' @param p Number of predictors.
#' @param n_signal Maximum number of nonzero coefficients.
#' @param scale Multiplicative signal strength.
#'
#' @return Numeric coefficient vector of length `p`.
#' @noRd
glmnet_fuzz_beta <- function(p, n_signal = 8L, scale = 0.7) {
  beta <- numeric(p)
  active <- seq_len(min(n_signal, p))
  beta[active] <- scale * seq(1, 0.35, length.out = length(active))
  beta
}

#' Generate difficult glmnet fuzz design matrices.
#'
#' @param n Number of observations.
#' @param p Number of predictors.
#' @param scenario Fuzz scenario name.
#' @param rho AR(1) correlation for non-block scenarios.
#'
#' @return Scaled numeric design matrix.
#' @noRd
glmnet_fuzz_design <- function(n, p, scenario, rho = 0.8) {
  if (scenario == "block_correlated") {
    block_size <- 5L
    x <- matrix(rnorm(n * p), nrow = n)
    n_blocks <- p %/% block_size
    for (block in seq_len(n_blocks)) {
      cols <- ((block - 1L) * block_size + 1L):(block * block_size)
      latent <- rnorm(n)
      noise <- matrix(rnorm(n * length(cols), sd = 0.12), nrow = n)
      x[, cols] <- latent + noise
    }
    return(scale(x))
  }
  scale(make_design(n = n, p = p, rho = rho))
}

#' Generate a train/test glmnet fuzz case.
#'
#' @param seed Integer seed.
#' @param scenario Fuzz scenario name.
#' @param family Model family.
#' @param n Number of training observations.
#' @param p Number of predictors.
#' @param n_test Number of test observations.
#' @param rho AR(1) correlation for non-block scenarios.
#' @param noise Gaussian noise standard deviation.
#'
#' @return A list containing train/test data, truth, and oracle test
#'   predictions from the data-generating model.
#' @noRd
glmnet_fuzz_case <- function(
  seed,
  scenario = c(
    "ar1_correlated", "block_correlated", "null_signal",
    "strong_rare_binomial"
  ),
  family = c("gaussian", "binomial"),
  n,
  p,
  n_test = n,
  rho = 0.8,
  noise = 1
) {
  scenario <- match.arg(scenario)
  family <- match.arg(family)
  set.seed(seed)
  x <- glmnet_fuzz_design(n, p, scenario, rho = rho)
  x_test <- glmnet_fuzz_design(n_test, p, scenario, rho = rho)
  colnames(x) <- colnames(x_test) <- paste0("x", seq_len(p))

  if (scenario == "null_signal") {
    beta <- numeric(p)
  } else if (scenario == "strong_rare_binomial") {
    beta <- glmnet_fuzz_beta(p, n_signal = 8L, scale = 1.2)
  } else {
    beta <- glmnet_fuzz_beta(p, n_signal = 8L, scale = 0.7)
  }

  eta_no_intercept <- drop(x %*% beta)
  intercept <- if (scenario == "strong_rare_binomial") {
    qlogis(0.03) - mean(eta_no_intercept)
  } else {
    0.1
  }
  eta <- intercept + eta_no_intercept
  eta_test <- intercept + drop(x_test %*% beta)

  if (family == "gaussian") {
    y <- eta + rnorm(n, sd = noise)
    y_test <- eta_test + rnorm(n_test, sd = noise)
    oracle_test_pred <- eta_test
  } else {
    prob <- plogis(pmin(pmax(eta, -30), 30))
    prob_test <- plogis(pmin(pmax(eta_test, -30), 30))
    y <- rbinom(n, size = 1L, prob = prob)
    y_test <- rbinom(n_test, size = 1L, prob = prob_test)
    oracle_test_pred <- prob_test
  }

  list(
    x = x,
    y = y,
    x_test = x_test,
    y_test = y_test,
    beta = beta,
    intercept = intercept,
    oracle_test_pred = oracle_test_pred
  )
}

#' Fit mlxs_glmnet and glmnet on the same lambda path.
#'
#' @param case A glmnet fuzz case.
#' @param family Model family.
#' @param alpha Elastic-net mixing parameter.
#' @param nlambda Number of lambda values.
#' @param lambda_min_ratio Smallest lambda as fraction of lambda max.
#' @param maxit Maximum mlxs_glmnet iterations per lambda.
#' @param tol mlxs_glmnet convergence tolerance.
#'
#' @return A list with reference fit, MLX fit, and lambda vector.
#' @noRd
fit_glmnet_pair <- function(
  case,
  family,
  alpha = 1,
  nlambda = 20L,
  lambda_min_ratio = 1e-3,
  maxit = 5000L,
  tol = 1e-7
) {
  ref <- glmnet::glmnet(
    case$x,
    case$y,
    family = family,
    alpha = alpha,
    nlambda = nlambda,
    lambda.min.ratio = lambda_min_ratio,
    standardize = FALSE,
    intercept = TRUE,
    thresh = 1e-12,
    maxit = 100000L
  )
  lambda <- as.numeric(ref$lambda)
  mlx_family <- if (family == "gaussian") {
    mlxs_gaussian()
  } else {
    mlxs_binomial()
  }
  mlx <- mlxs_glmnet(
    case$x,
    case$y,
    family = mlx_family,
    alpha = alpha,
    lambda = lambda,
    standardize = FALSE,
    intercept = TRUE,
    maxit = maxit,
    tol = tol
  )
  list(ref = ref, mlx = mlx, lambda = lambda)
}
