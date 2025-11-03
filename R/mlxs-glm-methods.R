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
#' @param formula,data Optional formula and data overrides used by
#'   `augment.mlxs_glm()`.
#' @param type.predict,type.residuals Character strings controlling the scale of
#'   fitted values and residuals returned by `augment.mlxs_glm()`.
#' @param se_fit Logical; standard-error analogue for `augment`.
#'
#' @name mlxs_glm_methods
NULL

.mlxs_family_as_base <- function(family) {
  fam_name <- family$family
  link_name <- family$link
  factory <- switch(
    fam_name,
    binomial = stats::binomial,
    quasibinomial = stats::quasibinomial,
    poisson = stats::poisson,
    quasipoisson = stats::quasipoisson,
    gaussian = stats::gaussian,
    Gamma = stats::Gamma,
    inverse.gaussian = stats::inverse.gaussian,
    NULL
  )
  if (is.null(factory)) {
    return(family)
  }
  factory(link = link_name)
}

.mlxs_glm_as_glm <- function(object) {
  X <- stats::model.matrix(object$terms, object$model)
  ww <- object$working.weights
  ww_sqrt <- sqrt(pmax(ww, .Machine$double.eps))
  qr <- base::qr(X * ww_sqrt)
  glm_call <- object$call
  if (!is.null(glm_call)) {
    glm_call[[1]] <- quote(stats::glm)
  }

  glm_obj <- list(
    coefficients = object$coefficients,
    residuals = object$deviance.resid,
    fitted.values = object$fitted.values,
    effects = NULL,
    R = NULL,
    rank = object$rank,
    family = .mlxs_family_as_base(object$family),
    linear.predictors = object$linear.predictors,
    deviance = object$deviance,
    aic = object$aic,
    null.deviance = object$null.deviance,
    iter = object$iter,
    weights = object$prior.weights,
    prior.weights = object$prior.weights,
    working.weights = object$working.weights,
    working.residuals = object$working.residuals,
    y = object$y,
    converged = object$converged,
    boundary = FALSE,
    df.residual = object$df.residual,
    df.null = object$df.null,
    dispersion = object$dispersion,
    call = glm_call,
    formula = stats::formula(object$terms),
    terms = object$terms,
    data = object$model,
    model = object$model,
    offset = object$offset,
    control = object$control,
    method = "mlxs_glm",
    contrasts = object$contrasts,
    xlevels = object$xlevels,
    na.action = object$na.action,
    qr = qr
  )
  class(glm_obj) <- c("glm", "lm")
  glm_obj
}

#' @rdname mlxs_glm_methods
#' @export
coef.mlxs_glm <- function(object, ...) {
  object$coefficients
}

#' @rdname mlxs_glm_methods
#' @export
predict.mlxs_glm <- function(object, newdata = NULL,
                              type = c("link", "response"),
                              se.fit = FALSE, ...) {
  type <- match.arg(type)
  glm_obj <- .mlxs_glm_as_glm(object)
  stats::predict(glm_obj, newdata = newdata, type = type, se.fit = se.fit, ...)
}

#' @rdname mlxs_glm_methods
#' @export
fitted.mlxs_glm <- function(object, ...) {
  object$fitted.values
}

#' @rdname mlxs_glm_methods
#' @export
residuals.mlxs_glm <- function(object,
                               type = c("deviance", "pearson", "working", "response"),
                               ...) {
  type <- match.arg(type)
  glm_obj <- .mlxs_glm_as_glm(object)
  stats::residuals(glm_obj, type = type, ...)
}

.mlxs_glm_vcov <- function(object) {
  qr_fit <- object$mlx$qr
  if (is.null(qr_fit)) {
    return(stats::vcov(.mlxs_glm_as_glm(object)))
  }
  r_mlx <- qr_fit$R
  n_coef <- length(object$coefficients)
  identity_mlx <- Rmlx::mlx_eye(n_coef)
  r_inv <- Rmlx::mlx_solve_triangular(r_mlx, identity_mlx, upper = TRUE)
  vcov_mlx <- r_inv %*% t(r_inv)
  vc <- as.matrix(vcov_mlx) * object$dispersion
  dimnames(vc) <- list(names(object$coefficients), names(object$coefficients))
  vc
}

#' @rdname mlxs_glm_methods
#' @export
vcov.mlxs_glm <- function(object, ...) {
  .mlxs_glm_vcov(object)
}

#' @rdname mlxs_glm_methods
#' @export
summary.mlxs_glm <- function(object, ...) {
  sum_glm <- summary(.mlxs_glm_as_glm(object), dispersion = object$dispersion, ...)
  class(sum_glm) <- c("summary.mlxs_glm", setdiff(class(sum_glm), "summary.mlxs_glm"))
  sum_glm
}

#' @rdname mlxs_glm_methods
#' @export
print.summary.mlxs_glm <- function(x, ...) {
  NextMethod("print")
}

#' @rdname mlxs_glm_methods
#' @export
anova.mlxs_glm <- function(object, ...) {
  convert <- function(obj) {
    if (inherits(obj, "mlxs_glm")) {
      stats::glm(
        formula = stats::formula(obj$terms),
        family = obj$family,
        data = obj$model
      )
    } else {
      obj
    }
  }
  glm_objects <- c(list(convert(object)), lapply(list(...), convert))
  do.call(stats::anova, glm_objects)
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
  length(object$y)
}

#' @rdname mlxs_glm_methods
#' @export
tidy.mlxs_glm <- function(x, ...) {
  sum_obj <- summary(x)
  coef_df <- sum_obj$coefficients
  statistic_col <- if ("z value" %in% colnames(coef_df)) "z value" else "t value"
  data.frame(
    term = rownames(coef_df),
    estimate = coef_df[, "Estimate"],
    std.error = coef_df[, "Std. Error"],
    statistic = coef_df[, statistic_col],
    p.value = coef_df[, ncol(coef_df)],
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
    nobs = length(x$y),
    converged = x$converged,
    iterations = x$iter,
    row.names = NULL
  )
}

#' @rdname mlxs_glm_methods
#' @export
augment.mlxs_glm <- function(x, data = x$model, newdata = NULL,
                             type.predict = c("response", "link"),
                             type.residuals = c("response", "deviance"),
                             se_fit = FALSE, ...) {
  type.predict <- match.arg(type.predict)
  type.residuals <- match.arg(type.residuals)
  if (se_fit) {
    stop("Standard errors for predictions are not implemented.", call. = FALSE)
  }

  if (is.null(newdata)) {
    out <- as.data.frame(data)
    out$.fitted <- predict(x, type = type.predict)
    out$.resid <- residuals(x, type = type.residuals)
  } else {
    out <- as.data.frame(newdata)
    out$.fitted <- predict(x, newdata = newdata, type = type.predict)
  }
  out
}
