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

# Helper to refit as base lm for operations that expect an lm object
.mlxs_as_lm <- function(object) {
  if (inherits(object, "lm")) {
    return(object)
  }
  if (!inherits(object, "mlxs_lm")) {
    stop("Expected an mlxs_lm or lm object.", call. = FALSE)
  }
  lm(terms(object), data = model.frame(object))
}

# Helper to compute variance-covariance matrix
.mlxs_vcov <- function(object) {
  qr_fit <- object$mlx$qr
  if (is.null(qr_fit)) {
    stop("QR decomposition not stored in mlxs_lm object.", call. = FALSE)
  }
  r_mlx <- qr_fit$R
  dim_r <- r_mlx$dim
  if (length(dim_r) != 2L || dim_r[1L] != dim_r[2L]) {
    stop("QR decomposition returned a non-square R matrix.", call. = FALSE)
  }

  eye <- Rmlx::mlx_eye(dim_r[1L])
  r_inv <- Rmlx::mlx_solve_triangular(r_mlx, eye, upper = TRUE)
  vcov_mlx <- r_inv %*% t(r_inv)

  residual_mlx <- object$mlx$residual
  if (!is.null(residual_mlx)) {
    rss <- drop(as.matrix(crossprod(residual_mlx)))
  } else {
    rss <- sum(object$residuals^2)
  }
  sigma2 <- rss / object$df.residual

  vcov_mlx <- vcov_mlx * sigma2
  vc <- as.matrix(vcov_mlx)
  colnames(vc) <- rownames(vc) <- names(object$coefficients)
  vc
}

#' @export
coef.mlxs_lm <- function(object, ...) {
  object$coefficients
}

#' @export
predict.mlxs_lm <- function(object, newdata = NULL, ...) {
  if (is.null(newdata)) {
    return(object$fitted.values)
  }
  terms_obj <- terms(object)
  mf <- model.frame(delete.response(terms_obj), data = newdata, na.action = na.pass)
  mm <- model.matrix(delete.response(terms_obj), mf)
  beta_mlx <- Rmlx::as_mlx(matrix(object$coefficients, ncol = 1))
  mm_mlx <- Rmlx::as_mlx(mm)
  preds <- mm_mlx %*% beta_mlx
  pred_vec <- drop(as.matrix(preds))
  if (!is.null(rownames(mm))) {
    names(pred_vec) <- rownames(mm)
  }
  pred_vec
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
  .mlxs_vcov(object)
}

#' @export
confint.mlxs_lm <- function(object, parm, level = 0.95, ...) {
  cf <- coef(object)
  if (missing(parm)) {
    parm <- seq_along(cf)
  } else if (is.character(parm)) {
    parm <- match(parm, names(cf), nomatch = NA_integer_)
    if (any(is.na(parm))) {
      stop("Some parameters not found in the model.", call. = FALSE)
    }
  }
  vc <- vcov(object)
  se <- sqrt(diag(vc))[parm]
  est <- cf[parm]
  alpha <- (1 - level) / 2
  t_quant <- qt(c(alpha, 1 - alpha), df = object$df.residual)
  limits <- outer(se, t_quant, `*`)
  ci <- cbind(est + limits[, 1], est + limits[, 2])
  probs <- c(alpha, 1 - alpha) * 100
  colnames(ci) <- paste0(sprintf("%g", probs), " %")
  rownames(ci) <- names(est)
  ci
}

#' @export
anova.mlxs_lm <- function(object, ...) {
  others <- list(...)
  lm_models <- c(list(.mlxs_as_lm(object)), lapply(others, .mlxs_as_lm))
  do.call(anova, lm_models)
}

#' @export
summary.mlxs_lm <- function(object, ...) {
  vc <- vcov(object)
  se <- sqrt(diag(vc))
  est <- coef(object)
  tvals <- est / se
  pvals <- 2 * pt(-abs(tvals), df = object$df.residual)
  coef_table <- cbind(Estimate = est, `Std. Error` = se, `t value` = tvals, `Pr(>|t|)` = pvals)

  resid <- residuals(object)
  rdf <- object$df.residual
  rss <- sum(resid^2)
  sigma <- sqrt(rss / rdf)
  response <- model.response(model.frame(object))
  tss <- sum((response - mean(response))^2)
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
    residuals = resid,
    coefficients = coef_table,
    sigma = sigma,
    df = c(object$rank, rdf, length(response)),
    r.squared = r.squared,
    adj.r.squared = if (rdf > 0) 1 - (1 - r.squared) * (length(response) - 1) / rdf else NA_real_,
    fstatistic = c(value = fstat, numdf = df_model, dendf = rdf),
    p.value = p_f,
    vcov = vc
  )
  class(result) <- c("summary.mlxs_lm", "mlxs_lm_summary")
  result
}

#' @export
print.summary.mlxs_lm <- function(x, ...) {
  cat("Call:\n")
  print(x$call)
  cat("\nResiduals:\n")
  resid_quants <- quantile(x$residuals, probs = c(0, 0.25, 0.5, 0.75, 1))
  names(resid_quants) <- c("Min", "1Q", "Median", "3Q", "Max")
  print(resid_quants)
  cat("\nCoefficients:\n")
  printCoefmat(x$coefficients, has.Pvalue = TRUE)
  cat("\nResidual standard error:", format(signif(x$sigma, 4)), "on", x$df[2], "degrees of freedom\n")
  if (!is.na(x$fstatistic[1])) {
    cat("Multiple R-squared:", format(signif(x$r.squared, 4)), ",  Adjusted R-squared:",
        format(signif(x$adj.r.squared, 4)), "\n")
    cat("F-statistic:", format(signif(x$fstatistic[1], 4)), "on", x$fstatistic[2], "and", x$fstatistic[3],
        "DF,  p-value:", format.pval(x$p.value), "\n")
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
  sum_obj <- summary(x)
  coef_df <- sum_obj$coefficients
  data.frame(
    term = rownames(coef_df),
    estimate = coef_df[, "Estimate"],
    std.error = coef_df[, "Std. Error"],
    statistic = coef_df[, "t value"],
    p.value = coef_df[, "Pr(>|t|)"],
    row.names = NULL
  )
}

#' @export
glance.mlxs_lm <- function(x, ...) {
  sum_obj <- summary(x)
  n <- nobs(x)
  rss <- sum(residuals(x)^2)
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
    p.value = sum_obj$p.value,
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
augment.mlxs_lm <- function(x, data = model.frame(x), newdata = NULL, se_fit = FALSE, ...) {
  terms_obj <- terms(x)
  if (is.null(newdata)) {
    mm <- model.matrix(x)
    fitted_vals <- x$fitted.values
    residuals_vals <- x$residuals
    base_data <- data
  } else {
    mf <- model.frame(delete.response(terms_obj), data = newdata, na.action = na.pass)
    mm <- model.matrix(delete.response(terms_obj), mf)
    beta_mlx <- Rmlx::as_mlx(matrix(x$coefficients, ncol = 1))
    mm_mlx <- Rmlx::as_mlx(mm)
    fitted_vals <- drop(as.matrix(mm_mlx %*% beta_mlx))
    residuals_vals <- NULL
    base_data <- newdata
  }

  if (!is.null(rownames(mm))) {
    names(fitted_vals) <- rownames(mm)
    if (!is.null(residuals_vals)) {
      names(residuals_vals) <- rownames(mm)
    }
  }

  out <- as.data.frame(base_data)
  out$.fitted <- fitted_vals
  if (is.null(newdata)) {
    out$.resid <- residuals_vals
  }

  if (se_fit) {
    vc <- vcov(x)
    se_vals <- sqrt(rowSums((mm %*% vc) * mm))
    if (!is.null(rownames(mm))) {
      names(se_vals) <- rownames(mm)
    }
    out$.se.fit <- se_vals
  }

  if (!is.null(rownames(mm))) {
    rownames(out) <- rownames(mm)
  }

  out
}
