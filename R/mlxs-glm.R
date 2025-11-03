#' MLX-backed generalized linear model
#'
#' Fit generalized linear models using iterative reweighted least squares (IRLS)
#' with MLX providing the heavy lifting for weighted least squares solves.
#'
#' @inheritParams stats::glm
#' @param family A mlxs family object (e.g., [mlxs_gaussian()], [mlxs_binomial()],
#'   [mlxs_poisson()]).
#' @param control Optional list of control parameters passed to
#'   [stats::glm.control()].
#'
#' @return An object of class `c("mlxs_glm", "mlxs_model")` containing elements
#'   similar to the result of [stats::glm()]. MLX intermediates are stored in the
#'   `mlx` field for downstream reuse. Computations use single-precision MLX
#'   tensors, so results typically agree with [stats::glm()] to around 1e-6
#'   unless a tighter tolerance is supplied via `control`.
#' @export
#'
#' @examples
#' fit <- mlxs_glm(mpg ~ cyl + disp, family = mlxs_gaussian(), data = mtcars)
#' coef(fit)
mlxs_glm <- function(formula, family = mlxs_gaussian(), data, subset,
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
  .as_mlx_col <- function(x) {
    Rmlx::as_mlx(matrix(x, ncol = 1))
  }

  X_mlx <- Rmlx::as_mlx(X)
  y_mlx <- .as_mlx_col(y)
  weights_mlx <- .as_mlx_col(weights)
  offset_mlx <- .as_mlx_col(offset)

  beta_mlx <- .as_mlx_col(coef_start)
  eta_mlx <- X_mlx %*% beta_mlx
  if (!all(offset == 0)) {
    eta_mlx <- eta_mlx + offset_mlx
  }
  mu_mlx <- family$linkinv(eta_mlx)
  mu_mlx <- .mlxs_glm_clamp_mu(mu_mlx, family)
  eta <- as.numeric(as.matrix(eta_mlx))
  mu <- as.numeric(as.matrix(mu_mlx))

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
      mu_mlx <- .as_mlx_col(mu)
    }
    if (exists("etastart", envir = env, inherits = FALSE)) {
      eta_candidate <- get("etastart", envir = env, inherits = FALSE)
      if (!is.null(eta_candidate)) {
        eta <- eta_candidate
        eta_mlx <- .as_mlx_col(eta)
      } else {
        eta_mlx <- family$linkfun(mu_mlx)
      }
    } else {
      eta_mlx <- family$linkfun(mu_mlx)
    }
  } else {
    mu_mlx <- family$linkinv(eta_mlx)
      mu <- as.numeric(as.matrix(mu_mlx))
  }

  eta <- as.numeric(as.matrix(eta_mlx))
  mu <- as.numeric(as.matrix(mu_mlx))

  dev_prev <- Inf
  converged <- FALSE
  mlx_state <- NULL
  iter_count <- control$maxit
  eps_mlx <- Rmlx::as_mlx(.Machine$double.eps)
  epsilon_target <- max(control$epsilon, 1e-6)

  mu_eta_mlx <- family$mu.eta(eta_mlx)
  w_mlx <- mu_eta_mlx / sqrt(family$variance(mu_mlx))

  for (iter in seq_len(control$maxit)) {
    var_mu_mlx <- family$variance(mu_mlx)
    if (any(!is.finite(as.numeric(as.matrix(var_mu_mlx))))) {
      stop("Non-finite variance function result.", call. = FALSE)
    }

    mu_eta_mlx <- family$mu.eta(eta_mlx)
    if (any(as.numeric(as.matrix(mu_eta_mlx)) == 0)) {
      stop("Zero derivative of link function detected.", call. = FALSE)
    }

    z_mlx <- eta_mlx + (y_mlx - mu_mlx) / mu_eta_mlx
    w_mlx <- mu_eta_mlx / sqrt(var_mu_mlx)
    w_mlx <- Rmlx::mlx_where(w_mlx < eps_mlx, eps_mlx, w_mlx)

    x_w_mlx <- Rmlx::mlx_broadcast_to(w_mlx, dim(X_mlx)) * X_mlx
    z_w_mlx <- z_mlx * w_mlx

    wls_fit <- .mlxs_wls(x_w_mlx, z_w_mlx)
    beta_new_mlx <- wls_fit$coefficients
    delta_val <- max(abs(as.numeric(as.matrix(beta_new_mlx - beta_mlx))))

    eta_mlx <- X_mlx %*% beta_new_mlx
    if (!all(offset == 0)) {
      eta_mlx <- eta_mlx + offset_mlx
    }
    mu_mlx <- family$linkinv(eta_mlx)
    mu_mlx <- .mlxs_glm_clamp_mu(mu_mlx, family)

    dev_res_mlx <- family$dev.resids(y_mlx, mu_mlx, weights_mlx)
    deviance_val <- sum(as.numeric(as.matrix(dev_res_mlx)))
    dev_change_val <- abs(deviance_val - dev_prev) / (0.1 + abs(deviance_val))

    if (control$trace) {
      message(
        "Iter ", iter, ": deviance = ", format(deviance_val, digits = 6),
        ", delta = ", format(delta_val, digits = 6),
        ", dev_change = ", format(dev_change_val, digits = 6)
      )
    }

    if (delta_val < epsilon_target || dev_change_val < epsilon_target) {
      converged <- TRUE
      mlx_state <- wls_fit$mlx
      beta_mlx <- beta_new_mlx
      dev_prev <- deviance_val
      iter_count <- iter
      break
    }
    if (!is.finite(deviance_val) || (deviance_val > dev_prev && abs(deviance_val - dev_prev) > epsilon_target)) {
      warning("Divergence detected in mlxs_glm; stopping iterations.", call. = FALSE)
      beta_mlx <- beta_new_mlx
      mlx_state <- wls_fit$mlx
      dev_prev <- deviance_val
      iter_count <- iter
      break
    }

    beta_mlx <- beta_new_mlx
    dev_prev <- deviance_val
    mlx_state <- wls_fit$mlx
    iter_count <- iter
  }

  if (!converged) {
    warning("mlxs_glm did not converge within maxit iterations.", call. = FALSE)
  }

  beta <- as.numeric(as.matrix(beta_mlx))
  eta <- as.numeric(as.matrix(eta_mlx))
  mu <- as.numeric(as.matrix(mu_mlx))
  w <- as.numeric(as.matrix(w_mlx))
  mu_eta_val <- as.numeric(as.matrix(mu_eta_mlx))

  dev_res_mlx <- family$dev.resids(y_mlx, mu_mlx, weights_mlx)
  dev_res_vec <- as.numeric(as.matrix(dev_res_mlx))
  deviance <- sum(dev_res_vec)

  fitted_values <- mu
  residuals <- y - fitted_values
  deviance_residuals <- sign(residuals) * sqrt(dev_res_vec)
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
  null_mu_mlx <- .as_mlx_col(null_mu)
  null_dev <- sum(as.numeric(as.matrix(family$dev.resids(y_mlx, null_mu_mlx, weights_mlx))))

  aic <- family$aic(y, weights, fitted_values, weights, deviance) + 2 * n_coef

  eta_mlx_out <- eta_mlx
  fitted_mlx <- mu_mlx
  resid_mlx <- y_mlx - mu_mlx

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
      eta = eta_mlx_out,
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


.mlxs_wls <- function(x_mlx, z_mlx) {
  qr_fit <- qr(x_mlx)
  qty <- crossprod(qr_fit$Q, z_mlx)
  coef_mlx <- Rmlx::mlx_solve_triangular(qr_fit$R, qty, upper = TRUE)
  fitted_mlx <- x_mlx %*% coef_mlx
  residual_mlx <- z_mlx - fitted_mlx

  list(
    coefficients = coef_mlx,
    fitted.values = fitted_mlx,
    residuals = residual_mlx,
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
    eps <- 1e-6
    eps_mlx <- Rmlx::as_mlx(eps)
    upper_mlx <- Rmlx::as_mlx(1 - eps)
    mu <- Rmlx::mlx_where(mu < eps_mlx, eps_mlx, mu)
    mu <- Rmlx::mlx_where(mu > upper_mlx, upper_mlx, mu)
  } else if (fam %in% c("poisson", "quasipoisson")) {
    eps <- 1e-6
    eps_mlx <- Rmlx::as_mlx(eps)
    mu <- Rmlx::mlx_where(mu < eps_mlx, eps_mlx, mu)
  }
  mu
}
