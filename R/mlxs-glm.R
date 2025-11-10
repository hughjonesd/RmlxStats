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
                     weights, na.action, start = NULL, control = list(), ...) {
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
  arg_names <- c("formula", "data", "subset", "weights", "na.action")
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

  weights_raw <- mf[["(weights)", exact = TRUE]]
  weights_supplied <- !is.null(weights_raw)
  weights_mlx <- if (is.null(weights_raw)) {
    Rmlx::mlx_full(c(n_obs, 1L), 1)
  } else if (inherits(weights_raw, "mlx")) {
    weights_raw
  } else {
    Rmlx::mlx_matrix(weights_raw, ncol = 1)
  }
  if (prod(Rmlx::mlx_dim(weights_mlx)) != n_obs) {
    stop("Length of 'weights' must match number of observations.", call. = FALSE)
  }
  if (as.logical(drop(as.matrix(Rmlx::mlx_any(!Rmlx::mlx_isfinite(weights_mlx)))))) {
    stop("Weights must be non-negative and finite.", call. = FALSE)
  }
  if (as.logical(drop(as.matrix(Rmlx::mlx_any(weights_mlx < 0))))) {
    stop("Weights must be non-negative and finite.", call. = FALSE)
  }
  offset <- rep.int(0, n_obs)
  offset_has_values <- any(offset != 0)

  X_mlx <- Rmlx::as_mlx(X)
  y_mlx <- if (inherits(y, "mlx")) y else Rmlx::mlx_matrix(y, ncol = 1)
  weights_sqrt_mlx <- sqrt(weights_mlx)
  offset_mlx <- if (offset_has_values) {
    Rmlx::mlx_matrix(offset, ncol = 1)
  } else {
    NULL
  }

  beta_mlx <- if (inherits(coef_start, "mlx")) coef_start else Rmlx::mlx_matrix(coef_start, ncol = 1)
  eta_mlx <- X_mlx %*% beta_mlx
  if (offset_has_values) {
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
      weights = .mlxs_as_numeric(weights_mlx),
      start = NULL,
      etastart = NULL,
      mustart = mu,
      offset = offset,
      nobs = n_obs,
      n = .mlxs_as_numeric(weights_mlx)
    )
    for (nm in names(initialize_vars)) {
      assign(nm, initialize_vars[[nm]], envir = env)
    }
    eval(family$initialize, envir = env)
    if (exists("mustart", envir = env, inherits = FALSE)) {
      mu <- get("mustart", envir = env, inherits = FALSE)
      mu_mlx <- Rmlx::mlx_matrix(mu, ncol = 1)
    }
    if (exists("etastart", envir = env, inherits = FALSE)) {
      eta_candidate <- get("etastart", envir = env, inherits = FALSE)
      if (!is.null(eta_candidate)) {
        eta <- eta_candidate
        eta_mlx <- Rmlx::mlx_matrix(eta, ncol = 1)
      } else {
        eta_mlx <- family$linkfun(mu_mlx)
      }
    } else {
      eta_mlx <- family$linkfun(mu_mlx)
    }
  } else {
    mu_mlx <- family$linkinv(eta_mlx)
    mu <- .mlxs_as_numeric(mu_mlx)
  }

  eps_mlx <- Rmlx::mlx_scalar(.Machine$double.eps)
  epsilon_target <- max(control$epsilon, 1e-6)
  compile_step <- isTRUE(getOption("mlxs.glm.compile", TRUE))

  irls_state <- .mlxs_glm_run_irls(
    X_mlx = X_mlx,
    y_mlx = y_mlx,
    family = family,
    weights_mlx = weights_mlx,
    weights_sqrt_mlx = weights_sqrt_mlx,
    offset_mlx = offset_mlx,
    beta_init = beta_mlx,
    eta_init = eta_mlx,
    mu_init = mu_mlx,
    control = control,
    eps_mlx = eps_mlx,
    epsilon_target = epsilon_target,
    trace = control$trace,
    compile_step = compile_step
  )

  beta_mlx <- irls_state$beta
  eta_mlx <- irls_state$eta
  mu_mlx <- irls_state$mu
  w_mlx <- irls_state$w
  mu_eta_mlx <- irls_state$mu_eta
  dev_res_mlx <- irls_state$dev_resids
  resid_mlx <- irls_state$residual
  iter_count <- irls_state$iter
  converged <- irls_state$converged
  deviance <- irls_state$deviance

  if (!converged) {
    warning("mlxs_glm did not converge within maxit iterations.", call. = FALSE)
  }

  beta <- .mlxs_as_numeric(beta_mlx)
  eta <- .mlxs_as_numeric(eta_mlx)
  mu <- .mlxs_as_numeric(mu_mlx)
  dev_res_vec <- .mlxs_as_numeric(dev_res_mlx)
  deviance <- sum(dev_res_vec)

  fitted_values <- mu
  residuals <- .mlxs_as_numeric(resid_mlx)
  deviance_resid_mlx <- sign(resid_mlx) * sqrt(dev_res_mlx)
  working_weights_mlx <- Rmlx::mlx_clip(w_mlx, min = .Machine$double.eps)^2
  working_residuals_mlx <- resid_mlx / mu_eta_mlx
  offset_result <- if (offset_has_values) offset_mlx else NULL
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
  null_mu_mlx <- Rmlx::mlx_matrix(null_mu, ncol = 1)
  null_dev <- sum(as.numeric(as.matrix(family$dev.resids(y_mlx, null_mu_mlx, weights_mlx))))

  weights_for_aic <- .mlxs_as_numeric(weights_mlx)
  aic <- family$aic(y, weights_for_aic, fitted_values, weights_for_aic, deviance) + 2 * n_coef

  result <- list(
    coefficients = beta_mlx,
    residuals = resid_mlx,
    fitted.values = mu_mlx,
    effects = NULL,
    rank = n_coef,
    family = family,
    linear.predictors = eta_mlx,
    deviance = deviance,
    aic = aic,
    null.deviance = null_dev,
    iter = iter_count,
    df.residual = df_residual,
    df.null = df_null,
    dispersion = dispersion,
    y = y_mlx,
    call = call,
    terms = terms,
    model = mf,
    deviance.resid = deviance_resid_mlx,
    converged = converged,
    weights = if (weights_supplied) weights_mlx else NULL,
    prior.weights = weights_mlx,
    working.weights = working_weights_mlx,
    working.residuals = working_residuals_mlx,
    mu_eta = mu_eta_mlx,
    offset = offset_result,
    coef_names = colnames(X),
    contrasts = attr(X, "contrasts"),
    xlevels = attr(mf, "xlevels"),
    na.action = attr(mf, "na.action"),
    control = control,
    qr = irls_state$qr
  )

  class(result) <- c("mlxs_glm", "mlxs_model")
  result
}
.mlxs_glm_clamp_mu <- function(mu, family) {
  fam <- family$family
  if (fam %in% c("binomial", "quasibinomial")) {
    eps <- 1e-6
    mu <- Rmlx::mlx_clip(mu, min = eps, max = 1 - eps)
  } else if (fam %in% c("poisson", "quasipoisson")) {
    eps <- 1e-6
    mu <- Rmlx::mlx_clip(mu, min = eps)
  }
  mu
}

.mlxs_glm_weighted_inputs_impl <- function(X_mlx,
                                           y_mlx,
                                           eta_mlx,
                                           mu_mlx,
                                           mu_eta_mlx,
                                           var_mu_mlx,
                                           weights_sqrt_mlx) {
  z_mlx <- eta_mlx + (y_mlx - mu_mlx) / mu_eta_mlx
  w_mlx <- mu_eta_mlx / sqrt(var_mu_mlx)
  if (!is.null(weights_sqrt_mlx)) {
    w_mlx <- w_mlx * weights_sqrt_mlx
  }
  w_mlx <- Rmlx::mlx_clip(w_mlx, min = .Machine$double.eps)
  dims <- Rmlx::mlx_dim(X_mlx)
  x_w_mlx <- Rmlx::mlx_broadcast_to(w_mlx, dims) * X_mlx
  z_w_mlx <- z_mlx * w_mlx
  list(z = z_mlx, w = w_mlx, x_w = x_w_mlx, z_w = z_w_mlx)
}

.mlxs_glm_weighted_inputs_runner <- local({
  compiled <- NULL
  warned <- FALSE
  function(..., compile = TRUE) {
    if (!compile) {
      return(.mlxs_glm_weighted_inputs_impl(...))
    }
    if (is.null(compiled)) {
      compiled <<- tryCatch(
        Rmlx::mlx_compile(.mlxs_glm_weighted_inputs_impl),
        error = function(e) {
          if (!warned) {
            warning("Falling back to uncompiled GLM step: ", e$message, call. = FALSE)
            warned <<- TRUE
          }
          FALSE
        }
      )
    }
    if (identical(compiled, FALSE)) {
      return(.mlxs_glm_weighted_inputs_impl(...))
    }
    out <- compiled(...)
    if (is.null(names(out))) {
      names(out) <- c("z", "w", "x_w", "z_w")
    }
    out
  }
})

.mlxs_glm_run_irls <- function(X_mlx,
                               y_mlx,
                               family,
                               weights_mlx,
                               weights_sqrt_mlx,
                               offset_mlx,
                               beta_init,
                               eta_init,
                               mu_init,
                               control,
                               eps_mlx,
                               epsilon_target,
                               trace = FALSE,
                               compile_step = TRUE) {
  beta_mlx <- beta_init
  eta_mlx <- eta_init
  mu_mlx <- mu_init
  mu_eta_mlx <- family$mu.eta(eta_mlx)
  w_mlx <- mu_eta_mlx / sqrt(family$variance(mu_mlx))
  if (!is.null(weights_sqrt_mlx)) {
    w_mlx <- w_mlx * weights_sqrt_mlx
  }
  dev_res_mlx <- family$dev.resids(y_mlx, mu_mlx, weights_mlx)
  dev_prev <- Inf
  qr_state <- NULL
  converged <- FALSE
  iter_count <- control$maxit

  for (iter in seq_len(control$maxit)) {
    var_mu_mlx <- family$variance(mu_mlx)
    var_numeric <- .mlxs_as_numeric(var_mu_mlx)
    if (any(!is.finite(var_numeric))) {
      stop("Non-finite variance function result.", call. = FALSE)
    }

    mu_eta_mlx <- family$mu.eta(eta_mlx)
    mu_eta_numeric <- .mlxs_as_numeric(mu_eta_mlx)
    if (any(mu_eta_numeric == 0)) {
      stop("Zero derivative of link function detected.", call. = FALSE)
    }

    step_inputs <- .mlxs_glm_weighted_inputs_runner(
      X_mlx, y_mlx, eta_mlx, mu_mlx, mu_eta_mlx, var_mu_mlx,
      weights_sqrt_mlx,
      compile = compile_step
    )
    w_mlx <- step_inputs$w

    wls_fit <- mlxs_lm_fit(step_inputs$x_w, step_inputs$z_w)
    beta_new_mlx <- wls_fit$coefficients
    qr_state <- wls_fit$qr
    delta_vec <- beta_new_mlx - beta_mlx
    delta_val <- max(abs(.mlxs_as_numeric(delta_vec)))

    eta_mlx <- X_mlx %*% beta_new_mlx
    if (!is.null(offset_mlx)) {
      eta_mlx <- eta_mlx + offset_mlx
    }
    mu_mlx <- family$linkinv(eta_mlx)
    mu_mlx <- .mlxs_glm_clamp_mu(mu_mlx, family)

    dev_res_mlx <- family$dev.resids(y_mlx, mu_mlx, weights_mlx)
    deviance_val <- sum(.mlxs_as_numeric(dev_res_mlx))
    dev_change_val <- if (is.finite(dev_prev)) {
      abs(deviance_val - dev_prev) / (0.1 + abs(deviance_val))
    } else {
      Inf
    }

    if (trace) {
      message(
        "Iter ", iter, ": deviance = ", format(deviance_val, digits = 6),
        ", delta = ", format(delta_val, digits = 6),
        ", dev_change = ", format(dev_change_val, digits = 6)
      )
    }

    if (delta_val < epsilon_target || dev_change_val < epsilon_target) {
      converged <- TRUE
      beta_mlx <- beta_new_mlx
      dev_prev <- deviance_val
      iter_count <- iter
      break
    }
    if (!is.finite(deviance_val) || (is.finite(dev_prev) && deviance_val > dev_prev && abs(deviance_val - dev_prev) > epsilon_target)) {
      warning("Divergence detected in mlxs_glm; stopping iterations.", call. = FALSE)
      beta_mlx <- beta_new_mlx
      dev_prev <- deviance_val
      iter_count <- iter
      break
    }

    beta_mlx <- beta_new_mlx
    dev_prev <- deviance_val
    iter_count <- iter
  }

  list(
    beta = beta_mlx,
    eta = eta_mlx,
    mu = mu_mlx,
    w = w_mlx,
    mu_eta = mu_eta_mlx,
    dev_resids = dev_res_mlx,
    residual = y_mlx - mu_mlx,
    deviance = dev_prev,
    iter = iter_count,
    converged = converged,
    qr = qr_state
  )
}
