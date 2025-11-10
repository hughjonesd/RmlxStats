#' mlxs_lm method utilities
#'
#' These helpers provide the familiar S3 surface for `mlxs_lm` fits.
#'
#' @name mlxs_lm_methods
#' @importFrom stats model.frame model.matrix model.response delete.response terms
#' @importFrom stats update.default predict fitted residuals nobs lm anova confint
#' @importFrom stats qt pf pt coef complete.cases na.pass quantile printCoefmat
#' @importFrom stats vcov
#' @importFrom generics tidy glance augment
NULL

.mlxs_coef_names <- function(object) {
  if (!is.null(object$coef_names)) {
    return(object$coef_names)
  }
  mm <- model.matrix(object$terms, model.frame(object))
  colnames(mm)
}

# Helper to refit as base lm for operations that expect an lm object
#' @export
coef.mlxs_lm <- function(object, ...) {
  coef_mlx <- object$coefficients
  attr(coef_mlx, "coef_names") <- .mlxs_coef_names(object)
  coef_mlx
}

#' @export
predict.mlxs_lm <- function(object, newdata = NULL, ...) {
  if (is.null(newdata)) {
    return(object$fitted.values)
  }
  terms_obj <- terms(object)
  mf <- model.frame(delete.response(terms_obj), data = newdata, na.action = na.pass)
  mm <- model.matrix(delete.response(terms_obj), mf)
  beta_mlx <- object$coefficients
  mm_mlx <- Rmlx::as_mlx(mm)
  preds <- mm_mlx %*% beta_mlx
  preds
}

#' @export
fitted.mlxs_lm <- function(object, ...) {
  object$fitted.values
}

#' @export
residuals.mlxs_lm <- function(object, ...) {
  object$residuals
}

#' @export
vcov.mlxs_lm <- function(object, ...) {
  qr_fit <- object$qr
  if (is.null(qr_fit)) {
    stop("QR decomposition not stored in mlxs_lm object.", call. = FALSE)
  }
  r_mlx <- qr_fit$R
  n_coef <- length(.mlxs_coef_names(object))
  eye <- Rmlx::mlx_eye(n_coef)
  r_inv <- Rmlx::mlx_solve_triangular(r_mlx, eye, upper = TRUE)
  sigma2 <- sum(.mlxs_as_numeric(object$residuals)^2) / object$df.residual
  sigma2 * (r_inv %*% t(r_inv))
}

#' @export
confint.mlxs_lm <- function(object, parm, level = 0.95, ...) {
  cf <- coef(object)
  cf_num <- .mlxs_as_numeric(cf)
  coef_names <- .mlxs_coef_names(object)
  if (missing(parm)) {
    parm <- seq_along(cf_num)
  } else if (is.character(parm)) {
    parm <- match(parm, coef_names, nomatch = NA_integer_)
    if (any(is.na(parm))) {
      stop("Some parameters not found in the model.", call. = FALSE)
    }
  }
  vc <- vcov(object)
  vc_mat <- as.matrix(vc)
  se <- sqrt(diag(vc_mat))[parm]
  est <- cf_num[parm]
  alpha <- (1 - level) / 2
  t_quant <- qt(c(alpha, 1 - alpha), df = object$df.residual)
  limits <- outer(se, t_quant, `*`)
  ci <- cbind(est + limits[, 1], est + limits[, 2])
  probs <- c(alpha, 1 - alpha) * 100
  colnames(ci) <- paste0(sprintf("%g", probs), " %")
  rownames(ci) <- coef_names[parm]
  ci
}

#' @export
anova.mlxs_lm <- function(object, ...) {
  stop("anova.mlxs_lm() is not implemented without converting to base lm.", call. = FALSE)
}

#' @export
summary.mlxs_lm <- function(object,
                            bootstrap = FALSE,
                            bootstrap_args = list(),
                            ...) {
  default_args <- list(
    B = 200L,
    seed = NULL,
    progress = FALSE,
    bootstrap_type = "case",
    batch_size = 32L
  )
  if (!is.list(bootstrap_args)) {
    stop("bootstrap_args must be a list.", call. = FALSE)
  }
  user_args <- utils::modifyList(default_args, bootstrap_args)
  bootstrap_type <- match.arg(user_args$bootstrap_type, c("case", "residual"))
  vc <- vcov(object)
  vc_mat <- as.matrix(vc)
  se <- sqrt(diag(vc_mat))
  bootstrap_info <- NULL
  if (isTRUE(bootstrap)) {
    bootstrap_info <- .mlxs_bootstrap_coefs(
      object,
      fit_type = "lm",
      B = user_args$B,
      seed = user_args$seed,
      progress = user_args$progress,
      batch_size = user_args$batch_size,
      method = bootstrap_type
    )
    se <- bootstrap_info$se
  }
  est <- .mlxs_as_numeric(object$coefficients)
  tvals <- est / se
  pvals <- 2 * pt(-abs(tvals), df = object$df.residual)
  resid_mlx <- residuals(object)
  rdf <- object$df.residual
  rss <- .mlxs_as_numeric(Rmlx::mlx_sum(resid_mlx * resid_mlx))
  sigma <- sqrt(rss / rdf)
  fitted_mlx <- fitted(object)
  y_mlx <- resid_mlx + fitted_mlx
  n_obs <- nobs(object)
  y_mean <- .mlxs_as_numeric(Rmlx::mlx_sum(y_mlx)) / n_obs
  centered <- y_mlx - Rmlx::mlx_scalar(y_mean)
  tss <- .mlxs_as_numeric(Rmlx::mlx_sum(centered * centered))
  r.squared <- if (tss < .Machine$double.eps) 1 else 1 - rss / tss
  df.int <- attr(object$terms, "intercept")
  if (is.null(df.int)) df.int <- 1L
  df_model <- object$rank - df.int
  if (df_model > 0) {
    ms_model <- (tss - rss) / df_model
    ms_error <- rss / rdf
    fstat <- ms_model / ms_error
    p_f <- pf(fstat, df_model, rdf, lower.tail = FALSE)
  } else {
    fstat <- NA_real_
    p_f <- NA_real_
  }

  result <- list(
    call = object$call,
    terms = object$terms,
    residuals = resid_mlx,
    coef = object$coefficients,
    coef_names = .mlxs_coef_names(object),
    std.error = Rmlx::mlx_matrix(se, ncol = 1),
    statistic = Rmlx::mlx_matrix(tvals, ncol = 1),
    p.value = Rmlx::mlx_matrix(pvals, ncol = 1),
    sigma = sigma,
    df = c(object$rank, rdf, n_obs),
    r.squared = r.squared,
    adj.r.squared = if (rdf > 0) 1 - (1 - r.squared) * (n_obs - 1) / rdf else NA_real_,
    fstatistic = c(value = fstat, numdf = df_model, dendf = rdf),
    p.value.model = p_f,
    cov.scaled = vc,
    cov.unscaled = vc / (sigma^2),
    bootstrap = bootstrap_info
  )
  class(result) <- c("summary.mlxs_lm", "mlxs_lm_summary")
  result
}

#' @export
print.mlxs_lm <- function(x, ...) {
  sum_obj <- summary(x, ...)
  print.summary.mlxs_lm(sum_obj, ...)
  invisible(x)
}

#' @export
print.summary.mlxs_lm <- function(x, ...) {
  cat("Call:\n")
  print(x$call)
  cat("\nResiduals:\n")
  resid_vals <- .mlxs_as_numeric(x$residuals)
  resid_quants <- quantile(resid_vals, probs = c(0, 0.25, 0.5, 0.75, 1))
  names(resid_quants) <- c("Min", "1Q", "Median", "3Q", "Max")
  print(resid_quants)
  cat("\nCoefficients:\n")
  coef_table <- cbind(
    Estimate = .mlxs_as_numeric(x$coef),
    `Std. Error` = .mlxs_as_numeric(x$std.error),
    `t value` = .mlxs_as_numeric(x$statistic),
    `Pr(>|t|)` = .mlxs_as_numeric(x$p.value)
  )
  rownames(coef_table) <- x$coef_names
  printCoefmat(coef_table, has.Pvalue = TRUE)
  cat("\nResidual standard error:", format(signif(x$sigma, 4)), "on", x$df[2], "degrees of freedom\n")
  if (!is.na(x$fstatistic[1])) {
    cat("Multiple R-squared:", format(signif(x$r.squared, 4)), ",  Adjusted R-squared:",
        format(signif(x$adj.r.squared, 4)), "\n")
    cat("F-statistic:", format(signif(x$fstatistic[1], 4)), "on", x$fstatistic[2], "and", x$fstatistic[3],
        "DF,  p-value:", format.pval(x$p.value.model), "\n")
  }
  invisible(x)
}

#' @export
update.mlxs_lm <- function(object, ..., evaluate = TRUE) {
  update.default(object, ..., evaluate = evaluate)
}

#' @export
model.frame.mlxs_lm <- function(formula, ...) {
  object <- formula
  mf <- object$model
  if (!is.null(mf) && is.null(attr(mf, "terms"))) {
    attr(mf, "terms") <- object$terms
  }
  mf
}

#' @export
model.matrix.mlxs_lm <- function(object, ...) {
  model.matrix(object$terms, model.frame(object), ...)
}

#' @export
terms.mlxs_lm <- function(x, ...) {
  x$terms
}

#' @export
nobs.mlxs_lm <- function(object, ...) {
  nrow(model.frame(object))
}

#' @export
tidy.mlxs_lm <- function(x, ...) {
  sum_obj <- summary(x, ...)
  data.frame(
    term = sum_obj$coef_names,
    estimate = .mlxs_as_numeric(sum_obj$coef),
    std.error = .mlxs_as_numeric(sum_obj$std.error),
    statistic = .mlxs_as_numeric(sum_obj$statistic),
    p.value = .mlxs_as_numeric(sum_obj$p.value),
    row.names = NULL
  )
}

#' @export
glance.mlxs_lm <- function(x, ...) {
  sum_obj <- summary(x, ...)
  n <- nobs(x)
  resid_vec <- .mlxs_as_numeric(residuals(x))
  rss <- sum(resid_vec^2)
  sigma <- sum_obj$sigma
  k <- sum_obj$df[1]
  loglik <- -0.5 * n * (log(2 * pi) + log(rss / n) + 1)
  aic <- -2 * loglik + 2 * k
  bic <- -2 * loglik + log(n) * k
  data.frame(
    r.squared = sum_obj$r.squared,
    adj.r.squared = sum_obj$adj.r.squared,
    sigma = sigma,
    statistic = sum_obj$fstatistic[1],
    p.value = sum_obj$p.value.model,
    df = sum_obj$df[1],
    df.residual = sum_obj$df[2],
    logLik = loglik,
    AIC = aic,
    BIC = bic,
    nobs = n,
    row.names = NULL
  )
}

#' @export
augment.mlxs_lm <- function(x, data = model.frame(x), newdata = NULL,
                            se_fit = FALSE,
                            output = c("data.frame", "mlx"),
                            ...) {
  terms_obj <- terms(x)
  output <- match.arg(output)
  if (is.null(newdata)) {
    mm <- model.matrix(x)
    fitted_vals <- x$fitted.values
    residuals_vals <- x$residuals
    base_data <- data
  } else {
    mf <- model.frame(delete.response(terms_obj), data = newdata, na.action = na.pass)
    mm <- model.matrix(delete.response(terms_obj), mf)
    mm_mlx <- Rmlx::as_mlx(mm)
    fitted_vals <- mm_mlx %*% x$coefficients
    residuals_vals <- NULL
    base_data <- newdata
  }

  if (output == "mlx") {
    return(list(.fitted = fitted_vals, .resid = residuals_vals))
  }

  fitted_num <- .mlxs_as_numeric(fitted_vals)
  residuals_num <- if (!is.null(residuals_vals)) .mlxs_as_numeric(residuals_vals) else NULL

  if (!is.null(rownames(mm))) {
    names(fitted_num) <- rownames(mm)
    if (!is.null(residuals_num)) {
      names(residuals_num) <- rownames(mm)
    }
  }

  if (!is.null(rownames(mm))) {
    rownames(base_data) <- rownames(mm)
  }

  out <- as.data.frame(base_data)
  out$.fitted <- fitted_num
  if (is.null(newdata)) {
    out$.resid <- residuals_num
  }

  if (se_fit) {
    vc <- as.matrix(vcov(x))
    se_vals <- sqrt(rowSums((mm %*% vc) * mm))
    if (!is.null(rownames(mm))) {
      names(se_vals) <- rownames(mm)
    }
    out$.se.fit <- se_vals
  }

  out
}
