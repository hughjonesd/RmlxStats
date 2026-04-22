#' Cross-validated MLX elastic net regression
#'
#' Cross-validation wrapper around [mlxs_glmnet()] that mirrors the core
#' `glmnet::cv.glmnet()` workflow for the families currently supported by
#' `mlxs_glmnet()`.
#'
#' The full-data fit defines a master lambda path. Each fold is then refit on
#' the same lambda values and scored on its holdout set.
#'
#' Current limitations relative to `glmnet::cv.glmnet()`:
#'
#' * only Gaussian and binomial families are supported
#' * `weights`, `offset`, `alignment != "lambda"`, `grouped = FALSE`,
#'   `parallel = TRUE`, `relax = TRUE`, and non-zero `trace.it` are not
#'   implemented
#' * `type.measure = "auc"` and `type.measure = "C"` are not implemented
#'
#' @param x Numeric matrix of predictors (observations in rows).
#' @param y Numeric response vector.
#' @param weights Optional observation weights. Currently unsupported.
#' @param offset Optional offset. Currently unsupported.
#' @param lambda Optional decreasing lambda sequence. If `NULL`, the full-data
#'   fit chooses the path and the same path is reused inside each fold.
#' @param type.measure Loss used to score the holdout predictions.
#' @param nfolds Number of folds.
#' @param foldid Optional integer vector giving the fold assignment for each
#'   observation.
#' @param alignment Alignment mode. Only `"lambda"` is currently supported.
#' @param grouped Should cross-validation be aggregated fold-by-fold? Only
#'   `TRUE` is currently supported.
#' @param keep Should out-of-fold predictions be stored?
#' @param parallel Logical. Parallel refits are currently unsupported.
#' @param gamma,relax Relaxed fits are currently unsupported.
#' @param trace.it Progress tracing. Currently unsupported.
#' @param family MLX-aware family object, e.g. [mlxs_gaussian()] or
#'   [mlxs_binomial()].
#' @param ... Additional arguments passed to [mlxs_glmnet()], such as `alpha`,
#'   `nlambda`, `lambda_min_ratio`, `standardize`, `intercept`, `maxit`, and
#'   `tol`.
#' @return An object of class `mlxs_cv_glmnet`.
#' @export
mlxs_cv_glmnet <- function(x,
                           y,
                           weights = NULL,
                           offset = NULL,
                           lambda = NULL,
                           type.measure = c(
                             "default", "mse", "deviance", "class",
                             "mae", "auc", "C"
                           ),
                           nfolds = 10,
                           foldid = NULL,
                           alignment = c("lambda", "fraction"),
                           grouped = TRUE,
                           keep = FALSE,
                           parallel = FALSE,
                           gamma = c(0, 0.25, 0.5, 0.75, 1),
                           relax = FALSE,
                           trace.it = 0,
                           family = mlxs_gaussian(),
                           ...) {
  type.measure <- match.arg(type.measure)
  alignment <- match.arg(alignment)
  family_name <- family$family

  if (!is.null(weights)) {
    stop("weights are not implemented for mlxs_cv_glmnet().",
         call. = FALSE)
  }
  if (!is.null(offset)) {
    stop("offset is not implemented for mlxs_cv_glmnet().",
         call. = FALSE)
  }
  if (alignment != "lambda") {
    stop("mlxs_cv_glmnet() currently supports alignment = 'lambda' only.",
         call. = FALSE)
  }
  if (!isTRUE(grouped)) {
    stop("mlxs_cv_glmnet() currently supports grouped = TRUE only.",
         call. = FALSE)
  }
  if (isTRUE(parallel)) {
    stop("parallel = TRUE is not implemented for mlxs_cv_glmnet().",
         call. = FALSE)
  }
  if (isTRUE(relax)) {
    stop("relax = TRUE is not implemented for mlxs_cv_glmnet().",
         call. = FALSE)
  }
  if (length(gamma) != 5L || any(gamma != c(0, 0.25, 0.5, 0.75, 1))) {
    stop("Relaxed gamma paths are not implemented for mlxs_cv_glmnet().",
         call. = FALSE)
  }
  if (!identical(trace.it, 0) && !identical(trace.it, FALSE)) {
    stop("trace.it is not implemented for mlxs_cv_glmnet().",
         call. = FALSE)
  }

  type.measure <- .mlxs_cv_glmnet_normalize_measure(type.measure, family_name)

  x <- as.matrix(x)
  y <- as.numeric(y)
  n_obs <- nrow(x)
  if (n_obs != length(y)) {
    stop("x and y must have the same number of observations.", call. = FALSE)
  }

  foldid <- .mlxs_cv_glmnet_foldid(
    y = y,
    n_obs = n_obs,
    nfolds = nfolds,
    foldid = foldid,
    family_name = family_name
  )
  nfolds <- length(unique(foldid))

  fit_args <- c(list(
    x = x,
    y = y,
    family = family,
    lambda = lambda
  ), list(...))
  glmnet_fit <- do.call(mlxs_glmnet, fit_args)

  lambda_path <- glmnet_fit$lambda
  n_lambda <- length(lambda_path)
  cvraw <- matrix(NA_real_, nrow = nfolds, ncol = n_lambda)
  if (keep) {
    fit_preval <- matrix(NA_real_, nrow = n_obs, ncol = n_lambda)
    dimnames(fit_preval) <- list(
      rownames(x),
      paste0("s", seq_len(n_lambda) - 1L)
    )
  } else {
    fit_preval <- NULL
  }

  for (fold_idx in seq_len(nfolds)) {
    holdout <- foldid == fold_idx
    train <- !holdout

    fold_fit_args <- c(list(
      x = x[train, , drop = FALSE],
      y = y[train],
      family = family,
      lambda = lambda_path
    ), list(...))
    fold_fit <- do.call(mlxs_glmnet, fold_fit_args)

    fold_pred <- predict(
      fold_fit,
      newx = x[holdout, , drop = FALSE],
      type = "response"
    )
    fold_loss <- .mlxs_cv_glmnet_loss(
      y = y[holdout],
      pred = fold_pred,
      family_name = family_name,
      type.measure = type.measure
    )

    cvraw[fold_idx, ] <- colMeans(fold_loss)
    if (keep) {
      fit_preval[holdout, ] <- fold_pred
    }
  }

  cvm <- colMeans(cvraw)
  cvsd <- apply(cvraw, 2L, stats::sd) / sqrt(nfolds)
  cvup <- cvm + cvsd
  cvlo <- cvm - cvsd

  min_idx <- which.min(cvm)
  one_se_idx <- which(cvm <= cvm[min_idx] + cvsd[min_idx])[1L]

  result <- list(
    lambda = lambda_path,
    cvm = cvm,
    cvsd = cvsd,
    cvup = cvup,
    cvlo = cvlo,
    nzero = colSums(abs(glmnet_fit$beta) > 0),
    call = match.call(),
    name = stats::setNames(
      .mlxs_cv_glmnet_measure_name(type.measure, family_name),
      type.measure
    ),
    glmnet.fit = glmnet_fit,
    lambda.min = lambda_path[min_idx],
    lambda.1se = lambda_path[one_se_idx],
    index = structure(
      matrix(c(min_idx, one_se_idx), nrow = 2L, ncol = 1L),
      dimnames = list(c("min", "1se"), "Lambda")
    ),
    foldid = foldid,
    type.measure = type.measure,
    grouped = grouped,
    cvraw = cvraw,
    fit.preval = fit_preval
  )

  if (!is.null(fit_preval) && family_name %in% c("binomial", "quasibinomial")) {
    attr(result$fit.preval, "classnames") <- sort(unique(as.character(y)))
  }

  class(result) <- "mlxs_cv_glmnet"
  result
}

.mlxs_cv_glmnet_normalize_measure <- function(type.measure, family_name) {
  if (type.measure == "default") {
    if (family_name == "gaussian") {
      return("mse")
    }
    if (family_name %in% c("binomial", "quasibinomial")) {
      return("deviance")
    }
  }

  if (type.measure %in% c("auc", "C")) {
    stop("type.measure = '", type.measure,
         "' is not implemented for mlxs_cv_glmnet().",
         call. = FALSE)
  }

  if (family_name == "gaussian" &&
      !type.measure %in% c("mse", "deviance", "mae")) {
    stop("Unsupported type.measure for gaussian family: ", type.measure,
         call. = FALSE)
  }

  if (family_name %in% c("binomial", "quasibinomial") &&
      !type.measure %in% c("mse", "deviance", "mae", "class")) {
    stop("Unsupported type.measure for binomial family: ", type.measure,
         call. = FALSE)
  }

  type.measure
}

.mlxs_cv_glmnet_measure_name <- function(type.measure, family_name) {
  if (type.measure == "mse") {
    return("Mean-Squared Error")
  }
  if (type.measure == "mae") {
    return("Mean Absolute Error")
  }
  if (type.measure == "class") {
    return("Misclassification Error")
  }
  if (family_name == "gaussian") {
    return("Mean-Squared Error")
  }
  "Binomial Deviance"
}

.mlxs_cv_glmnet_foldid <- function(y,
                                   n_obs,
                                   nfolds,
                                   foldid,
                                   family_name) {
  if (!is.null(foldid)) {
    foldid <- as.integer(foldid)
    if (length(foldid) != n_obs) {
      stop("foldid must have length nrow(x).", call. = FALSE)
    }
    if (anyNA(foldid)) {
      stop("foldid cannot contain NA values.", call. = FALSE)
    }
    uniq <- sort(unique(foldid))
    if (length(uniq) < 2L) {
      stop("foldid must define at least two folds.", call. = FALSE)
    }
    remap <- match(foldid, uniq)
    return(remap)
  }

  if (nfolds < 2L) {
    stop("nfolds must be at least 2.", call. = FALSE)
  }
  if (nfolds > n_obs) {
    stop("nfolds cannot exceed the number of observations.", call. = FALSE)
  }

  if (family_name %in% c("binomial", "quasibinomial")) {
    foldid <- integer(n_obs)
    for (level in c(0, 1)) {
      idx <- which(y == level)
      foldid[idx] <- sample(rep(seq_len(nfolds), length.out = length(idx)))
    }
    return(foldid)
  }

  sample(rep(seq_len(nfolds), length.out = n_obs))
}

.mlxs_cv_glmnet_loss <- function(y, pred, family_name, type.measure) {
  pred <- as.matrix(pred)

  if (family_name == "gaussian") {
    y_mat <- matrix(y, nrow = length(y), ncol = ncol(pred))
    if (type.measure %in% c("mse", "deviance")) {
      return((pred - y_mat)^2)
    }
    return(abs(pred - y_mat))
  }

  prob <- pmin(pmax(pred, 1e-8), 1 - 1e-8)
  y_mat <- matrix(y, nrow = length(y), ncol = ncol(prob))

  switch(
    type.measure,
    deviance = -2 * (y_mat * log(prob) + (1 - y_mat) * log(1 - prob)),
    mse = (y_mat - prob)^2,
    mae = abs(y_mat - prob),
    class = (prob >= 0.5) != (y_mat >= 0.5)
  )
}

.mlxs_glmnet_select_path <- function(object, s = NULL, exact = FALSE) {
  if (isTRUE(exact)) {
    stop("exact = TRUE is not implemented for mlxs_glmnet methods.",
         call. = FALSE)
  }

  beta <- as.matrix(object$beta)
  a0 <- as.numeric(object$a0)
  lambda <- as.numeric(object$lambda)

  if (is.null(s)) {
    return(list(
      beta = beta,
      a0 = a0,
      lambda = lambda,
      names = paste0("s", seq_along(lambda) - 1L)
    ))
  }

  s_names <- names(s)
  s <- as.numeric(s)
  if (anyNA(s)) {
    stop("s cannot contain NA values.", call. = FALSE)
  }

  log_lambda <- log(lambda)
  log_s <- log(s)
  n_pred <- nrow(beta)
  beta_out <- matrix(NA_real_, nrow = n_pred, ncol = length(s))
  a0_out <- numeric(length(s))

  for (j in seq_along(s)) {
    if (log_s[j] >= log_lambda[1L]) {
      beta_out[, j] <- beta[, 1L]
      a0_out[j] <- a0[1L]
      next
    }
    if (log_s[j] <= log_lambda[length(log_lambda)]) {
      beta_out[, j] <- beta[, length(log_lambda)]
      a0_out[j] <- a0[length(log_lambda)]
      next
    }

    right <- which(log_lambda <= log_s[j])[1L]
    left <- right - 1L
    if (isTRUE(all.equal(log_s[j], log_lambda[left], tolerance = 1e-12))) {
      beta_out[, j] <- beta[, left]
      a0_out[j] <- a0[left]
      next
    }
    if (isTRUE(all.equal(log_s[j], log_lambda[right], tolerance = 1e-12))) {
      beta_out[, j] <- beta[, right]
      a0_out[j] <- a0[right]
      next
    }

    weight_left <- (log_s[j] - log_lambda[right]) /
      (log_lambda[left] - log_lambda[right])
    weight_right <- 1 - weight_left

    beta_out[, j] <- beta[, left] * weight_left + beta[, right] * weight_right
    a0_out[j] <- a0[left] * weight_left + a0[right] * weight_right
  }

  list(beta = beta_out, a0 = a0_out, lambda = s, names = s_names)
}

#' @export
coef.mlxs_glmnet <- function(object, s = NULL, exact = FALSE, ...) {
  selected <- .mlxs_glmnet_select_path(object, s = s, exact = exact)
  coef_mat <- rbind("(Intercept)" = selected$a0, selected$beta)
  beta_names <- rownames(object$beta)
  if (is.null(beta_names)) {
    beta_names <- paste0("V", seq_len(nrow(object$beta)))
  }
  rownames(coef_mat)[-1L] <- beta_names

  col_names <- selected$names
  if (is.null(col_names) || !length(col_names)) {
    col_names <- paste0("s", seq_len(ncol(coef_mat)) - 1L)
  }
  colnames(coef_mat) <- col_names
  coef_mat
}

#' @export
predict.mlxs_glmnet <- function(object,
                                newx,
                                s = NULL,
                                type = c(
                                  "link", "response", "coefficients",
                                  "nonzero", "class"
                                ),
                                exact = FALSE,
                                ...) {
  type <- match.arg(type)

  if (type == "coefficients") {
    return(coef(object, s = s, exact = exact, ...))
  }

  selected <- .mlxs_glmnet_select_path(object, s = s, exact = exact)
  if (type == "nonzero") {
    return(lapply(seq_len(ncol(selected$beta)), function(j) {
      which(abs(selected$beta[, j]) > 0)
    }))
  }

  newx <- as.matrix(newx)
  if (ncol(newx) != nrow(object$beta)) {
    stop("newx must have the same number of columns as the fitted x.",
         call. = FALSE)
  }

  eta <- newx %*% selected$beta +
    matrix(selected$a0, nrow = nrow(newx), ncol = length(selected$a0),
           byrow = TRUE)

  if (type == "link") {
    return(eta)
  }

  if (object$family == "gaussian") {
    return(eta)
  }

  response <- 1 / (1 + exp(-eta))
  if (type == "class") {
    return(ifelse(response >= 0.5, 1, 0))
  }
  response
}

.mlxs_cv_glmnet_resolve_s <- function(object, s) {
  if (is.numeric(s)) {
    return(s)
  }
  if (is.character(s)) {
    s <- match.arg(s, c("lambda.1se", "lambda.min"))
    lambda <- object[[s]]
    names(lambda) <- s
    return(lambda)
  }
  stop("Invalid form for s.", call. = FALSE)
}

#' @export
coef.mlxs_cv_glmnet <- function(object,
                                s = c("lambda.1se", "lambda.min"),
                                ...) {
  lambda <- .mlxs_cv_glmnet_resolve_s(object, s)
  coef(object$glmnet.fit, s = lambda, ...)
}

#' @export
predict.mlxs_cv_glmnet <- function(object,
                                   newx,
                                   s = c("lambda.1se", "lambda.min"),
                                   ...) {
  lambda <- .mlxs_cv_glmnet_resolve_s(object, s)
  predict(object$glmnet.fit, newx = newx, s = lambda, ...)
}

#' @export
print.mlxs_cv_glmnet <- function(x,
                                 digits = getOption("digits"),
                                 ...) {
  cat(
    "MLX cross-validated elastic net fit (family = ",
    x$glmnet.fit$family,
    ", measure = ",
    unname(x$name),
    ")\n",
    sep = ""
  )
  cat(
    "  lambda.min = ",
    format(signif(x$lambda.min, digits = digits)),
    ", lambda.1se = ",
    format(signif(x$lambda.1se, digits = digits)),
    "\n",
    sep = ""
  )
  invisible(x)
}
