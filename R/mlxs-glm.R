#' MLX-backed generalized linear model
#'
#' Fit generalized linear models using iterative reweighted least squares (IRLS)
#' with MLX providing the heavy lifting for weighted least squares solves.
#'
#' @inheritParams stats::glm
#' @param family A GLM family object or specification.
#' @param control Optional list of control parameters passed to
#'   [stats::glm.control()].
#'
#' @return An object of class `c("mlxs_glm", "mlxs_model")` containing elements
#'   similar to the result of [stats::glm()]. MLX intermediates are stored in the
#'   `mlx` field for downstream reuse.
#' @export
#'
#' @examples
#' if (requireNamespace("Rmlx", quietly = TRUE)) {
#'   fit <- mlxs_glm(mpg ~ cyl + disp, family = stats::gaussian(), data = mtcars)
#'   coef(fit)
#' }
mlxs_glm <- function(formula, family = stats::gaussian(), data, subset,
                     na.action, start = NULL, control = list(), ...) {
  call <- match.call()

  if (is.character(family)) {
    family <- get(family, mode = "function", envir = parent.frame())
  }
  if (is.function(family)) {
    family <- family(...)
  }
  if (is.null(family$variance) || is.null(family$mu.eta)) {
    stop("Invalid 'family' argument.", call. = FALSE)
  }

  control <- do.call(stats::glm.control, control)

  mf <- match.call(expand.dots = FALSE)
  arg_names <- c("formula", "data", "subset", "na.action")
  keep <- match(arg_names, names(mf), nomatch = 0L)
  mf <- mf[c(1L, keep)]
  mf$drop.unused.levels <- TRUE
  mf[[1L]] <- quote(stats::model.frame)
  mf <- eval(mf, parent.frame())

  terms <- attr(mf, "terms")
  y <- stats::model.response(mf)
  if (is.matrix(y) && ncol(y) == 1L) {
    y <- drop(y)
  }
  if (!is.numeric(y)) {
    y <- stats::model.response(mf, "numeric")
  }

  X <- stats::model.matrix(terms, mf)
  n_obs <- nrow(X)
  n_coef <- ncol(X)

  coef_start <- if (is.null(start)) rep.int(0, n_coef) else start

  weights <- rep.int(1, n_obs)
  offset <- rep.int(0, n_obs)
  eta <- drop(X %*% coef_start)
  mu <- family$linkinv(eta)
  mu <- .mlxs_glm_clamp_mu(mu, family)

  if (!is.null(family$initialize)) {
    env <- new.env(parent = environment())
    initialize_vars <- list(
      y = y,
      weights = weights,
      start = NULL,
      etastart = NULL,
      mustart = mu,
      offset = offset,
      nobs = n_obs,
      n = weights
    )
    for (nm in names(initialize_vars)) {
      assign(nm, initialize_vars[[nm]], envir = env)
    }
    eval(family$initialize, envir = env)
    if (exists("mustart", envir = env, inherits = FALSE)) {
      mu <- get("mustart", envir = env, inherits = FALSE)
    }
    if (exists("etastart", envir = env, inherits = FALSE)) {
      eta_candidate <- get("etastart", envir = env, inherits = FALSE)
      if (!is.null(eta_candidate)) {
        eta <- eta_candidate
      } else {
        eta <- family$linkfun(mu)
      }
    } else {
      eta <- family$linkfun(mu)
    }
  } else {
    mu <- family$linkinv(eta)
  }

  dev_prev <- Inf
  beta <- coef_start
  converged <- FALSE
  mlx_state <- NULL
  iter_count <- control$maxit

  for (iter in seq_len(control$maxit)) {
    var_mu <- family$variance(mu)
    mu_eta_val <- family$mu.eta(eta)
    if (any(!is.finite(var_mu))) {
      stop("Non-finite variance function result.", call. = FALSE)
    }
    if (any(mu_eta_val == 0)) {
      stop("Zero derivative of link function detected.", call. = FALSE)
    }

    z <- eta + (y - mu) / mu_eta_val
    w <- mu_eta_val / sqrt(var_mu)
    w <- pmax(w, .Machine$double.eps)

    x_w <- sweep(X, 1L, w, `*`)
    z_w <- z * w

    wls_fit <- .mlxs_wls(x_w, z_w)
    beta_new <- wls_fit$coefficients
    delta <- max(abs(beta_new - beta))

    eta <- drop(X %*% beta_new)
    mu <- family$linkinv(eta)
    mu <- .mlxs_glm_clamp_mu(mu, family)

    deviance <- sum(family$dev.resids(y, mu, weights))
    dev_change <- abs(deviance - dev_prev) / (0.1 + abs(deviance))

    if (control$trace) {
      message("Iter ", iter, ": deviance = ", format(deviance, digits = 6),
              ", delta = ", format(delta, digits = 6),
              ", dev_change = ", format(dev_change, digits = 6))
    }

    if (delta < control$epsilon || dev_change < control$epsilon) {
      converged <- TRUE
      mlx_state <- wls_fit$mlx
      beta <- beta_new
      iter_count <- iter
      break
    }
    if (!is.finite(deviance) || (deviance > dev_prev && abs(deviance - dev_prev) > control$epsilon)) {
      warning("Divergence detected in mlxs_glm; stopping iterations.", call. = FALSE)
      beta <- beta_new
      mlx_state <- wls_fit$mlx
      iter_count <- iter
      break
    }

    beta <- beta_new
    dev_prev <- deviance
    mlx_state <- wls_fit$mlx
    iter_count <- iter
  }

  if (!converged) {
    warning("mlxs_glm did not converge within maxit iterations.", call. = FALSE)
  }

  fitted_values <- mu
  residuals <- y - fitted_values
  deviance_residuals <- sign(residuals) * sqrt(family$dev.resids(y, fitted_values, weights))
  deviance <- sum(family$dev.resids(y, fitted_values, weights))
  working_weights <- pmax(w, .Machine$double.eps)^2
  working_residuals <- (y - fitted_values) / mu_eta_val

  if (!is.null(rownames(X))) {
    names(fitted_values) <- rownames(X)
    names(residuals) <- rownames(X)
    names(working_residuals) <- rownames(X)
    names(eta) <- rownames(X)
    names(weights) <- rownames(X)
    names(working_weights) <- rownames(X)
  }
  names(beta) <- colnames(X)
  intercept_present <- attr(terms, "intercept") > 0
  df_residual <- n_obs - n_coef
  df_null <- n_obs - as.integer(intercept_present)

  dispersion <- if (family$family %in% c("binomial", "poisson")) {
    1
  } else if (df_residual > 0) {
    deviance / df_residual
  } else {
    NA_real_
  }

  if (family$family %in% c("binomial", "quasibinomial")) {
    y_mean <- mean(y)
    y_mean <- min(max(y_mean, .Machine$double.eps), 1 - .Machine$double.eps)
    null_mu <- rep(y_mean, n_obs)
  } else {
    null_mu <- rep(mean(y), n_obs)
  }
  null_dev <- sum(family$dev.resids(y, null_mu, weights))

  aic <- family$aic(y, weights, fitted_values, weights, deviance) + 2 * n_coef

  eta_mlx <- Rmlx::as_mlx(matrix(eta, ncol = 1))
  fitted_mlx <- Rmlx::as_mlx(matrix(fitted_values, ncol = 1))
  resid_mlx <- Rmlx::as_mlx(matrix(residuals, ncol = 1))

  result <- list(
    coefficients = beta,
    residuals = residuals,
    fitted.values = fitted_values,
    effects = NULL,
    rank = n_coef,
    family = family,
    linear.predictors = eta,
    deviance = deviance,
    aic = aic,
    null.deviance = null_dev,
    iter = iter_count,
    df.residual = df_residual,
    df.null = df_null,
    dispersion = dispersion,
    y = y,
    call = call,
    terms = terms,
    model = mf,
    deviance.resid = deviance_residuals,
    mlx = list(
      qr = mlx_state$qr,
      x = mlx_state$x,
      y = mlx_state$y,
      residual = mlx_state$residual,
      coef = mlx_state$coef,
      eta = eta_mlx,
      fitted = fitted_mlx,
      residual_vector = resid_mlx
    ),
    converged = converged,
    weights = weights,
    prior.weights = weights,
    working.weights = working_weights,
    working.residuals = working_residuals,
    offset = offset,
    contrasts = attr(X, "contrasts"),
    xlevels = attr(mf, "xlevels"),
    na.action = attr(mf, "na.action"),
    control = control
  )

  class(result) <- c("mlxs_glm", "mlxs_model")
  result
}

.mlxs_wls <- function(x, z) {
  x_mlx <- Rmlx::as_mlx(x)
  z_mlx <- Rmlx::as_mlx(matrix(z, ncol = 1))

  qr_fit <- qr(x_mlx)
  qty <- crossprod(qr_fit$Q, z_mlx)
  coef_mlx <- Rmlx::mlx_solve_triangular(qr_fit$R, qty, upper = TRUE)
  fitted_mlx <- x_mlx %*% coef_mlx
  residual_mlx <- z_mlx - fitted_mlx

  list(
    coefficients = drop(as.matrix(coef_mlx)),
    fitted.values = drop(as.matrix(fitted_mlx)),
    residuals = drop(as.matrix(residual_mlx)),
    mlx = list(
      qr = qr_fit,
      x = x_mlx,
      y = z_mlx,
      residual = residual_mlx,
      coef = coef_mlx
    )
  )
}

.mlxs_glm_clamp_mu <- function(mu, family) {
  fam <- family$family
  if (fam %in% c("binomial", "quasibinomial")) {
    eps <- .Machine$double.eps
    mu <- pmin(pmax(mu, eps), 1 - eps)
  } else if (fam %in% c("poisson", "quasipoisson")) {
    mu <- pmax(mu, .Machine$double.eps)
  }
  mu
}
