#' mlxs_glm method utilities
#'
#' Support functions that provide a familiar S3 surface for `mlxs_glm`
#' fits by delegating to equivalent base `glm` behaviour where helpful.
#'
#' @param object An `mlxs_glm` model fit.
#' @param ... Additional arguments passed to underlying methods.
#' @param newdata Optional data frame used for prediction.
#' @param type Character string indicating the scale of the prediction or
#'   residuals to return.
#' @param se.fit Logical. Should standard errors of the fit be returned when
#'   supported?
#' @param x An `mlxs_glm` model fit (for methods with a leading `x` argument).
#' @param digits Number of significant digits to print for summaries.
#' @param formula,data Optional formula and data overrides used by
#'   `augment.mlxs_glm()`.
#' @param type.predict,type.residuals Character strings controlling the scale of
#'   fitted values and residuals returned by `augment.mlxs_glm()`.
#' @param se_fit Logical; standard-error analogue for `augment`.
#' @param bootstrap Logical; should bootstrap standard errors be computed?
#' @param bootstrap_args List of bootstrap configuration options.
#' @param output Character string; return format ("data.frame" or "mlx").
#'
#' @name mlxs_glm_methods
NULL

#' @rdname mlxs_glm_methods
#' @export
coef.mlxs_glm <- function(object, ...) {
  coef_mlx <- object$coefficients
  attr(coef_mlx, "coef_names") <- .mlxs_coef_names(object)
  coef_mlx
}

#' @rdname mlxs_glm_methods
#' @export
predict.mlxs_glm <- function(
  object,
  newdata = NULL,
  type = c("link", "response"),
  se.fit = FALSE,
  ...
) {
  type <- match.arg(type)
  if (isTRUE(se.fit)) {
    stop(
      "Prediction standard errors are not implemented for mlxs_glm.",
      call. = FALSE
    )
  }
  if (is.null(newdata)) {
    return(
      if (type == "response") object$fitted.values else object$linear.predictors
    )
  }

  terms_obj <- object$terms
  mf <- stats::model.frame(
    stats::delete.response(terms_obj),
    data = newdata,
    na.action = stats::na.pass,
    xlev = object$xlevels
  )
  mm <- stats::model.matrix(
    stats::delete.response(terms_obj),
    mf,
    contrasts.arg = object$contrasts
  )
  mm_mlx <- Rmlx::as_mlx(mm)
  eta <- mm_mlx %*% object$coefficients
  if (type == "response") {
    return(object$family$linkinv(eta))
  }
  eta
}

#' @rdname mlxs_glm_methods
#' @export
fitted.mlxs_glm <- function(object, ...) {
  object$fitted.values
}

#' @rdname mlxs_glm_methods
#' @export
residuals.mlxs_glm <- function(
  object,
  type = c("deviance", "pearson", "working", "response"),
  ...
) {
  type <- match.arg(type)
  if (type == "response") {
    return(object$residuals)
  }
  if (type == "deviance") {
    return(object$deviance.resid)
  }
  if (type == "working") {
    return(object$working.residuals)
  }

  y_mlx <- object$y
  mu_mlx <- object$fitted.values
  var_mu <- object$family$variance(mu_mlx)
  pearson <- (y_mlx - mu_mlx) / sqrt(var_mu)
  if (!is.null(object$prior.weights)) {
    pearson <- pearson * sqrt(object$prior.weights)
  }
  pearson
}

#' @rdname mlxs_glm_methods
#' @export
vcov.mlxs_glm <- function(object, ...) {
  qr_fit <- object$qr
  n_coef <- length(.mlxs_coef_names(object))
  .mlxs_vcov_from_qr(qr_fit, n_coef = n_coef, scale = object$dispersion)
}

#' @rdname mlxs_glm_methods
#' @export
print.mlxs_glm <- function(x, digits = max(3, getOption("digits") - 3), ...) {
  sum_obj <- summary(x, ...)
  print.summary.mlxs_glm(sum_obj, digits = digits, ...)
  invisible(x)
}

#' @rdname mlxs_glm_methods
#' @export
summary.mlxs_glm <- function(
  object,
  bootstrap = FALSE,
  bootstrap_args = list(),
  ...
) {
  default_args <- list(
    B = 200L,
    seed = NULL,
    progress = FALSE,
    bootstrap_type = "case"
  )
  if (!is.list(bootstrap_args)) {
    stop("bootstrap_args must be a list.", call. = FALSE)
  }
  user_args <- utils::modifyList(default_args, bootstrap_args)
  bootstrap_type <- match.arg(user_args$bootstrap_type, c("case", "residual"))

  coef_names <- .mlxs_coef_names(object)
  coef_mlx <- object$coefficients
  vcov_mlx <- vcov(object)
  diag_mlx <- Rmlx::diag(vcov_mlx)
  n_coef <- length(coef_names)
  se_col_mlx <- Rmlx::mlx_reshape(sqrt(diag_mlx), c(n_coef, 1L))

  bootstrap_info <- NULL
  if (isTRUE(bootstrap)) {
    bootstrap_info <- .mlxs_bootstrap_coefs(
      object,
      fit_type = "glm",
      B = user_args$B,
      seed = user_args$seed,
      progress = user_args$progress,
      method = bootstrap_type
    )
    se_col_mlx <- bootstrap_info$se
    se_sq_row <- Rmlx::mlx_reshape(se_col_mlx^2, c(1L, length(coef_names)))
    diag_eye <- Rmlx::mlx_eye(length(coef_names))
    vcov_mlx <- diag_eye *
      Rmlx::mlx_broadcast_to(se_sq_row, Rmlx::mlx_shape(diag_eye))
  }
  stat_mlx <- coef_mlx / se_col_mlx
  stat_label <- if (object$family$family %in% c("gaussian", "quasigaussian")) {
    "t value"
  } else {
    "z value"
  }

  p_mlx <- if (stat_label == "t value") {
    stat_num <- as.numeric(stat_mlx)
    p_num <- 2 * stats::pt(-abs(stat_num), df = object$df.residual)
    Rmlx::mlx_vector(p_num)
  } else {
    2 * Rmlx::mlx_pnorm(-abs(stat_mlx))
  }

  sum_list <- list(
    call = object$call,
    family = object$family,
    coef_names = coef_names,
    coefficients = coef_mlx,
    std.error = se_col_mlx,
    statistic = stat_mlx,
    p.value = p_mlx,
    stat_label = stat_label,
    dispersion = object$dispersion,
    df.residual = object$df.residual,
    df.null = object$df.null,
    null.deviance = object$null.deviance,
    deviance = object$deviance,
    aic = object$aic,
    deviance.resid = object$deviance.resid,
    residuals = object$residuals,
    working.residuals = object$working.residuals,
    cov.scaled = vcov_mlx,
    cov.unscaled = vcov_mlx / object$dispersion,
    bootstrap = bootstrap_info
  )
  class(sum_list) <- "summary.mlxs_glm"
  sum_list
}

#' @rdname mlxs_glm_methods
#' @export
print.summary.mlxs_glm <- function(
  x,
  digits = max(3, getOption("digits") - 3),
  ...
) {
  cat("Call:\n")
  print(x$call)
  est <- as.numeric(x$coefficients)
  se <- as.numeric(x$std.error)
  stat <- as.numeric(x$statistic)
  p <- as.numeric(x$p.value)
  stat_col <- x$stat_label
  p_col <- if (stat_col == "t value") "Pr(>|t|)" else "Pr(>|z|)"
  stat_block <- cbind(stat, p)
  colnames(stat_block) <- c(stat_col, p_col)
  coef_table <- cbind(
    Estimate = est,
    `Std. Error` = se,
    stat_block
  )
  rownames(coef_table) <- x$coef_names
  cat("\nCoefficients:\n")
  printCoefmat(coef_table, digits = digits, has.Pvalue = TRUE)
  cat(
    "\n(Dispersion parameter for",
    x$family$family,
    "family taken to be",
    format(signif(x$dispersion, digits)),
    ")\n"
  )
  cat(
    "Null deviance:",
    format(signif(x$null.deviance, digits)),
    "on",
    x$df.null,
    "degrees of freedom\n"
  )
  cat(
    "Residual deviance:",
    format(signif(x$deviance, digits)),
    "on",
    x$df.residual,
    "degrees of freedom\n"
  )
  cat("AIC:", format(signif(x$aic, digits)), "\n")
  if (!is.null(x$bootstrap)) {
    cat(
      "\nBootstrap standard errors (",
      x$bootstrap$B,
      " resamples) applied.\n",
      sep = ""
    )
  }
  invisible(x)
}

#' @rdname mlxs_glm_methods
#' @export
anova.mlxs_glm <- function(object, ...) {
  if (nargs() > 1L) {
    stop(
      "anova.mlxs_glm() does not yet compare multiple MLX models.",
      call. = FALSE
    )
  }
  stop(
    "anova.mlxs_glm() is not implemented without converting to base glm.",
    call. = FALSE
  )
}

#' @rdname mlxs_glm_methods
#' @export
model.frame.mlxs_glm <- function(formula, ...) {
  formula$model
}

#' @rdname mlxs_glm_methods
#' @export
model.matrix.mlxs_glm <- function(object, ...) {
  stats::model.matrix(object$terms, object$model, ...)
}

#' @rdname mlxs_glm_methods
#' @export
terms.mlxs_glm <- function(x, ...) {
  x$terms
}

#' @rdname mlxs_glm_methods
#' @export
nobs.mlxs_glm <- function(object, ...) {
  nrow(model.frame(object))
}

#' @rdname mlxs_glm_methods
#' @export
tidy.mlxs_glm <- function(x, ...) {
  sum_obj <- summary(x, ...)
  data.frame(
    term = sum_obj$coef_names,
    estimate = as.numeric(sum_obj$coefficients),
    std.error = as.numeric(sum_obj$std.error),
    statistic = as.numeric(sum_obj$statistic),
    p.value = as.numeric(sum_obj$p.value),
    row.names = NULL
  )
}

#' @rdname mlxs_glm_methods
#' @export
glance.mlxs_glm <- function(x, ...) {
  k <- x$rank
  loglik <- -0.5 * (x$aic - 2 * k)
  data.frame(
    aic = x$aic,
    deviance = x$deviance,
    null.deviance = x$null.deviance,
    df.residual = x$df.residual,
    df.null = x$df.null,
    logLik = loglik,
    nobs = nobs(x),
    converged = x$converged,
    iterations = x$iter,
    row.names = NULL
  )
}

#' @rdname mlxs_glm_methods
#' @export
augment.mlxs_glm <- function(
  x,
  data = x$model,
  newdata = NULL,
  type.predict = c("response", "link"),
  type.residuals = c("response", "deviance"),
  se_fit = FALSE,
  output = c("data.frame", "mlx"),
  ...
) {
  type.predict <- match.arg(type.predict)
  type.residuals <- match.arg(type.residuals)
  output <- match.arg(output)
  if (se_fit) {
    stop("Standard errors for predictions are not implemented.", call. = FALSE)
  }

  fitted_vals <- predict(x, newdata = newdata, type = type.predict)
  resid_vals <- if (is.null(newdata)) {
    residuals(x, type = type.residuals)
  } else {
    NULL
  }

  if (output == "mlx") {
    return(list(.fitted = fitted_vals, .resid = resid_vals))
  }

  out <- as.data.frame(if (is.null(newdata)) data else newdata)
  out$.fitted <- as.numeric(fitted_vals)
  if (!is.null(resid_vals)) {
    out$.resid <- as.numeric(resid_vals)
  }
  if (is.null(newdata)) {
    rownames(out) <- rownames(data)
  }
  out
}
