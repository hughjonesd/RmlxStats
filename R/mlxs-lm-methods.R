#' mlxs_lm method utilities
#'
#' These helpers provide the familiar S3 surface for `mlxs_lm` fits.
#'
#' @param object,model An `mlxs_lm` model fit.
#' @param x An `mlxs_lm` model fit (for methods with a leading `x` argument).
#' @param ... Additional arguments passed to underlying methods.
#' @param newdata Optional data frame for prediction.
#' @param parm Parameter specification for confidence intervals.
#' @param level Confidence level for intervals.
#' @param bootstrap Logical; should bootstrap standard errors or confidence
#'   intervals be computed?
#' @param bootstrap_args List of bootstrap configuration options. 
#'   See [mlxs_boot()].
#' @param confint Logical; should confidence intervals be included in the
#'   summary object?
#' @param formula An `mlxs_lm` object used in place of formula for
#'   `model.frame`.
#' @param data Optional data frame for `augment`.
#' @param se_fit Logical; should standard errors of fit be included?
#' @param output Character string; return format ("data.frame", "matrix", "vector" or "mlx").
#'   To make methods from other packages work, the usual default is to return a base
#'   R object.
#' @param row.names Optional row names for data frame conversion.
#' @param optional Logical; passed to `as.data.frame`.
#'
#' @name mlxs-lm-methods
#' @importFrom stats model.frame model.matrix model.response delete.response
#' @importFrom stats terms
#' @importFrom stats predict fitted residuals nobs lm anova confint
#' @importFrom stats qt pf pt coef complete.cases na.pass quantile printCoefmat
#' @importFrom stats vcov hatvalues naresid weights
NULL

#' @export
#' @rdname mlxs-lm-methods
coef.mlxs_lm <- function(object, ..., output = c("vector", "mlx")) {
  output <- match.arg(output)
  coef <- object$coefficients
  if (identical(output, "vector")) {
    coef <- as.numeric(coef) 
    names(coef) <- .mlxs_coef_names(object)
  }
  
  coef
}

#' @export
#' @rdname mlxs-lm-methods
predict.mlxs_lm <- function(object, newdata = NULL, ...) {
  if (is.null(newdata)) {
    return(.mlxs_napredict(object$na.action, object$fitted.values))
  }
  terms_obj <- terms(object)
  mf <- model.frame(
    delete.response(terms_obj),
    data = newdata,
    na.action = na.pass
  )
  mm <- model.matrix(delete.response(terms_obj), mf)
  beta_mlx <- object$coefficients
  mm_mlx <- Rmlx::as_mlx(mm)
  preds <- mm_mlx %*% beta_mlx
  preds
}

#' @export
#' @rdname mlxs-lm-methods
fitted.mlxs_lm <- function(object, ...) {
  .mlxs_napredict(object$na.action, object$fitted.values)
}

#' @export
#' @rdname mlxs-lm-methods
residuals.mlxs_lm <- function(object, ...) {
  .mlxs_naresid(object$na.action, object$residuals)
}

#' @export
#' @rdname mlxs-lm-methods
vcov.mlxs_lm <- function(object, ...) {
  qr_fit <- object$qr
  n_coef <- length(.mlxs_coef_names(object))
  rss <- as.numeric(.mlxs_weighted_sum_of_squares(
    object$residuals,
    object$weights
  ))
  sigma2 <- rss / object$df.residual
  vcov_mlx <- .mlxs_vcov_from_qr(qr_fit, n_coef = n_coef, scale = sigma2)
  coef_names <- .mlxs_coef_names(object)
  dimnames(vcov_mlx) <- list(coef_names, coef_names)
  vcov_mlx
}

#' @export
#' @rdname mlxs-lm-methods
confint.mlxs_lm <- function(
  object,
  parm,
  level = 0.95,
  ...,
  bootstrap = FALSE,
  bootstrap_args = list(
    B = 200L,
    seed = NULL,
    progress = FALSE,
    bootstrap_type = "case"
  )
) {
  coef_names <- .mlxs_coef_names(object)
  if (isTRUE(bootstrap)) {
    names(bootstrap_args)[names(bootstrap_args) == "bootstrap_type"] <- "method"
    bootstrap_info <- do.call(
      .mlxs_bootstrap_coefs,
      c(list(object = object, fit_type = "lm", level = level), bootstrap_args)
    )
    return(bootstrap_info$confint[parm, , drop = FALSE])
  }
  cf <- coef(object)
  cf_num <- as.numeric(cf)
  vc <- vcov(object)
  se <- as.numeric(sqrt(Rmlx::diag(vc)))[parm]
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
#' @rdname mlxs-lm-methods
anova.mlxs_lm <- function(object, ...) {
  others <- list(...)
  if (length(others) > 0L) {
    stop(
      "anova.mlxs_lm() does not yet compare multiple mlxs_lm models.",
      call. = FALSE
    )
  }

  if (is.null(object$residuals) || is.null(object$fitted.values)) {
    stop(
      "Fitted values and residuals are required to compute ANOVA.",
      call. = FALSE
    )
  }

  assign_vec <- object$assign
  if (is.null(assign_vec)) {
    mm <- model.matrix(object$terms, object$model)
    assign_vec <- attr(mm, "assign")
  }
  if (is.null(assign_vec)) {
    stop("Unable to recover assign vector from the model.", call. = FALSE)
  }

  effects_mlx <- object$effects
  if (is.null(effects_mlx)) {
    stop(
      "QR effects are missing; refit the model to use anova().",
      call. = FALSE
    )
  }

  p <- if (!is.null(object$rank)) object$rank else 0L
  seq_idx <- if (p > 0L) seq_len(p) else integer()
  asgn_seq <- assign_vec[seq_idx]
  effects_seq <- if (p > 0L) effects_mlx[seq_idx, , drop = FALSE] else NULL

  group_ids <- unique(asgn_seq)
  group_rows <- lapply(group_ids, function(id) which(asgn_seq == id))
  df_terms_all <- lengths(group_rows, use.names = FALSE)
  term_labels <- attr(object$terms, "term.labels")
  if (is.null(term_labels)) {
    term_labels <- character()
  }
  label_map <- vapply(
    group_ids,
    function(id) {
      if (id == 0L) {
        "(Intercept)"
      } else if (id <= length(term_labels)) {
        term_labels[id]
      } else {
        paste0("term_", id)
      }
    },
    character(1)
  )
  sumsq_terms_all <- lapply(group_rows, function(rows) {
    comp_rows <- effects_seq[rows, , drop = FALSE]
    Rmlx::mlx_sum(comp_rows * comp_rows)
  })

  intercept_attr <- attr(object$terms, "intercept")
  intercept_present <- !is.null(intercept_attr) && intercept_attr != 0
  keep_idx <- seq_along(label_map)
  if (intercept_present) {
    keep_idx <- keep_idx[label_map != "(Intercept)"]
  }
  label_terms <- label_map[keep_idx]
  df_terms <- df_terms_all[keep_idx]
  sumsq_terms <- sumsq_terms_all[keep_idx]

  resid_ss <- .mlxs_weighted_sum_of_squares(object$residuals, object$weights)
  fitted_ss <- .mlxs_weighted_sum_of_squares(
    object$fitted.values,
    object$weights
  )
  if (
    as.numeric(resid_ss) <
      1e-10 * max(as.numeric(fitted_ss), .Machine$double.eps)
  ) {
    warning(
      "ANOVA F-tests on an essentially perfect fit are unreliable",
      call. = FALSE
    )
  }

  resid_df <- object$df.residual
  resid_ms <- resid_ss / Rmlx::mlx_scalar(resid_df)
  meansq_terms <- Map(
    function(ss, df) ss / Rmlx::mlx_scalar(df),
    sumsq_terms,
    df_terms
  )
  f_terms <- lapply(meansq_terms, function(ms) ms / resid_ms)
  f_numeric <- vapply(f_terms, as.numeric, numeric(1))
  p_vals <- pf(f_numeric, df_terms, resid_df, lower.tail = FALSE)

  result <- list(
    labels = c(label_terms, "Residuals"),
    df = c(df_terms, resid_df),
    sumsq = c(sumsq_terms, list(resid_ss)),
    meansq = c(meansq_terms, list(resid_ms)),
    fvalue = c(f_terms, list(Rmlx::mlx_scalar(NA_real_))),
    pvalue = c(p_vals, NA_real_)
  )
  vars <- attr(object$terms, "variables")
  response_label <- if (!is.null(vars) && length(vars) >= 2L) {
    paste(deparse(vars[[2L]], width.cutoff = 500L), collapse = "")
  } else {
    "<response>"
  }
  heading <- c(
    "Analysis of Variance Table\n",
    paste("Response:", response_label)
  )
  attr(result, "heading") <- heading
  class(result) <- c("mlxs_anova", "anova")
  result
}

#' @export
#' @rdname mlxs-lm-methods
as.data.frame.mlxs_anova <- function(
  x,
  row.names = NULL,
  optional = FALSE,
  ...
) {
  sumsq_num <- vapply(x$sumsq, as.numeric, numeric(1))
  meansq_num <- vapply(x$meansq, as.numeric, numeric(1))
  fvalue_num <- vapply(
    x$fvalue,
    function(val) {
      if (is.null(val)) {
        NA_real_
      } else {
        out <- as.numeric(val)
        if (length(out) == 0L || is.nan(out)) NA_real_ else out
      }
    },
    numeric(1)
  )

  table <- data.frame(
    Df = x$df,
    `Sum Sq` = sumsq_num,
    `Mean Sq` = meansq_num,
    `F value` = fvalue_num,
    `Pr(>F)` = x$pvalue,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  if (is.null(row.names)) {
    row.names <- x$labels
  }
  rownames(table) <- row.names
  attr(table, "heading") <- attr(x, "heading")
  class(table) <- c("anova", "data.frame")
  table
}

#' @export
#' @rdname mlxs-lm-methods
print.mlxs_anova <- function(x, ...) {
  df <- as.data.frame(x)
  print(df, ...)
  invisible(x)
}

#' @export
#' @rdname mlxs-lm-methods
tidy.mlxs_anova <- function(x, ...) {
  df <- as.data.frame(x)
  data.frame(
    term = rownames(df),
    df = df$Df,
    sumsq = df[["Sum Sq"]],
    meansq = df[["Mean Sq"]],
    statistic = df[["F value"]],
    p.value = df[["Pr(>F)"]],
    row.names = NULL,
    stringsAsFactors = FALSE
  )
}

#' @export
#' @rdname mlxs-lm-methods
summary.mlxs_lm <- function(
  object,
  bootstrap = FALSE,
  bootstrap_args = list(
    B = 200L,
    seed = NULL,
    progress = FALSE,
    bootstrap_type = "case"
  ),
  confint = FALSE,
  level = 0.95,
  ...
) {
  coef_names <- .mlxs_coef_names(object)
  vc <- vcov(object)
  vc_mat <- as.matrix(vc)
  se_mlx <- Rmlx::mlx_matrix(sqrt(diag(vc_mat)), ncol = 1)
  bootstrap_info <- NULL
  confint_mat <- NULL
  if (isTRUE(bootstrap)) {
    names(bootstrap_args)[names(bootstrap_args) == "bootstrap_type"] <- "method"
    bootstrap_info <- do.call(
      .mlxs_bootstrap_coefs,
      c(list(object = object, fit_type = "lm", level = level), bootstrap_args)
    )
    se_mlx <- bootstrap_info$se
    if (isTRUE(confint)) {
      confint_mat <- bootstrap_info$confint
    } else {
      bootstrap_info$confint <- NULL
    }
  } else if (isTRUE(confint)) {
    confint_mat <- stats::confint(object, level = level)
  }
  se_num <- as.numeric(se_mlx)
  est <- as.numeric(object$coefficients)
  tvals <- est / se_num
  pvals <- 2 * pt(-abs(tvals), df = object$df.residual)
  rownames(se_mlx) <- coef_names
  statistic_mlx <- Rmlx::mlx_matrix(tvals, ncol = 1)
  p_value_mlx <- Rmlx::mlx_matrix(pvals, ncol = 1)
  rownames(statistic_mlx) <- coef_names
  rownames(p_value_mlx) <- coef_names
  resid_mlx <- object$residuals
  rdf <- object$df.residual
  rss <- as.numeric(Rmlx::mlx_sum(resid_mlx * resid_mlx))
  sigma <- sqrt(rss / rdf)
  fitted_mlx <- object$fitted.values
  y_mlx <- resid_mlx + fitted_mlx
  n_obs <- nobs(object)
  y_mean <- as.numeric(Rmlx::mlx_sum(y_mlx)) / n_obs
  centered <- y_mlx - Rmlx::mlx_scalar(y_mean)
  tss <- as.numeric(Rmlx::mlx_sum(centered * centered))
  r.squared <- if (tss < .Machine$double.eps) 1 else 1 - rss / tss
  df.int <- attr(object$terms, "intercept")
  if (is.null(df.int)) {
    df.int <- 1L
  }
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
    std.error = se_mlx,
    statistic = statistic_mlx,
    p.value = p_value_mlx,
    sigma = sigma,
    df = c(object$rank, rdf, n_obs),
    r.squared = r.squared,
    adj.r.squared = if (rdf > 0) {
      1 - (1 - r.squared) * (n_obs - 1) / rdf
    } else {
      NA_real_
    },
    fstatistic = c(value = fstat, numdf = df_model, dendf = rdf),
    p.value.model = p_f,
    cov.scaled = vc,
    cov.unscaled = vc / (sigma^2),
    confint = confint_mat,
    bootstrap = bootstrap_info
  )
  class(result) <- c("summary.mlxs_lm", "mlxs_lm_summary")
  result
}

#' @export
#' @rdname mlxs-lm-methods
print.mlxs_lm <- function(x, ...) {
  sum_obj <- summary(x, ...)
  print.summary.mlxs_lm(sum_obj, ...)
  invisible(x)
}

.mlxs_summary_coef_table <- function(
  estimate,
  std_error,
  confint = NULL,
  statistic,
  p_value,
  stat_col,
  p_col
) {
  stat_block <- cbind(statistic, p_value)
  colnames(stat_block) <- c(stat_col, p_col)
  do.call(
    cbind,
    c(
      list(Estimate = estimate, `Std. Error` = std_error),
      if (!is.null(confint)) list(confint) else list(),
      list(stat_block)
    )
  )
}

#' @export
#' @rdname mlxs-lm-methods
print.summary.mlxs_lm <- function(x, ...) {
  cat("Call:\n")
  print(x$call)
  cat("\nResiduals:\n")
  resid_vals <- as.numeric(x$residuals)
  resid_quants <- quantile(resid_vals, probs = c(0, 0.25, 0.5, 0.75, 1))
  names(resid_quants) <- c("Min", "1Q", "Median", "3Q", "Max")
  print(resid_quants)
  cat("\nCoefficients:\n")
  coef_table <- .mlxs_summary_coef_table(
    estimate = as.numeric(x$coef),
    std_error = as.numeric(x$std.error),
    confint = x$confint,
    statistic = as.numeric(x$statistic),
    p_value = as.numeric(x$p.value),
    stat_col = "t value",
    p_col = "Pr(>|t|)"
  )
  rownames(coef_table) <- rownames(x$coef) %||% x$coef_names
  printCoefmat(coef_table, has.Pvalue = TRUE)
  cat(
    "\nResidual standard error:",
    format(signif(x$sigma, 4)),
    "on",
    x$df[2],
    "degrees of freedom\n"
  )
  if (!is.na(x$fstatistic[1])) {
    cat(
      "Multiple R-squared:",
      format(signif(x$r.squared, 4)),
      ",  Adjusted R-squared:",
      format(signif(x$adj.r.squared, 4)),
      "\n"
    )
    cat(
      "F-statistic:",
      format(signif(x$fstatistic[1], 4)),
      "on",
      x$fstatistic[2],
      "and",
      x$fstatistic[3],
      "DF,  p-value:",
      format.pval(x$p.value.model),
      "\n"
    )
  }
  invisible(x)
}

#' @export
#' @rdname mlxs-lm-methods
model.frame.mlxs_lm <- function(formula, ...) {
  object <- formula
  mf <- object$model
  if (!is.null(mf) && is.null(attr(mf, "terms"))) {
    attr(mf, "terms") <- object$terms
  }
  mf
}

#' @export
#' @rdname mlxs-lm-methods
model.matrix.mlxs_lm <- function(object, ...) {
  model.matrix(object$terms, model.frame(object), ...)
}

#' @export
#' @rdname mlxs-lm-methods
terms.mlxs_lm <- function(x, ...) {
  x$terms
}

#' @export
#' @rdname mlxs-lm-methods
nobs.mlxs_lm <- function(object, ...) {
  nrow(model.frame(object))
}

#' @export
#' @rdname mlxs-lm-methods
tidy.mlxs_lm <- function(x, ...) {
  sum_obj <- summary(x, ...)
  coef_names <- rownames(sum_obj$coef) %||% sum_obj$coef_names
  data.frame(
    term = coef_names,
    estimate = as.numeric(sum_obj$coef),
    std.error = as.numeric(sum_obj$std.error),
    statistic = as.numeric(sum_obj$statistic),
    p.value = as.numeric(sum_obj$p.value),
    row.names = NULL
  )
}

#' @export
#' @rdname mlxs-lm-methods
glance.mlxs_lm <- function(x, ...) {
  sum_obj <- summary(x, ...)
  n <- nobs(x)
  resid_vec <- as.numeric(x$residuals)
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
#' @rdname mlxs-lm-methods
augment.mlxs_lm <- function(
  x,
  data = model.frame(x),
  newdata = NULL,
  se_fit = FALSE,
  output = c("data.frame", "mlx"),
  ...
) {
  terms_obj <- terms(x)
  output <- match.arg(output)
  if (is.null(newdata)) {
    mm <- model.matrix(x)
    fitted_vals <- x$fitted.values
    residuals_vals <- x$residuals
    base_data <- data
  } else {
    mf <- model.frame(
      delete.response(terms_obj),
      data = newdata,
      na.action = na.pass
    )
    mm <- model.matrix(delete.response(terms_obj), mf)
    mm_mlx <- Rmlx::as_mlx(mm)
    fitted_vals <- mm_mlx %*% x$coefficients
    residuals_vals <- NULL
    base_data <- newdata
  }

  if (output == "mlx") {
    return(list(.fitted = fitted_vals, .resid = residuals_vals))
  }

  fitted_num <- as.numeric(fitted_vals)
  residuals_num <- if (!is.null(residuals_vals)) {
    as.numeric(residuals_vals)
  } else {
    NULL
  }

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
