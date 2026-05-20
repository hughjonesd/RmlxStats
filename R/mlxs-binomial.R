#' MLX-friendly binomial family
#'
#' Construct a binomial GLM family whose core link and deviance helpers are
#' implemented in R so they work with MLX arrays as well as base R vectors.
#' This avoids calling into compiled C routines that only handle base types.
#'
#' Currently the `logit`, `log`, `cloglog`, and `cauchit` links are supported.
#' For other link specifications, fall back to [stats::binomial()].
#'
#' @inheritParams stats::binomial
#'
#' @return A family object compatible with [stats::glm()] and `mlxs_glm()`.
#' @export
mlxs_binomial <- function(link = "logit") {
  base_family <- stats::binomial(link = link)
  link_name <- base_family$link

  link_parts <- .mlxs_binomial_link(link_name)
  if (is.null(link_parts)) {
    warning(
      sprintf(
        "Link '%s' is not currently MLX-optimised; using stats::binomial().",
        link_name
      ),
      call. = FALSE
    )
    return(base_family)
  }

  base_family$linkfun <- link_parts$linkfun
  base_family$linkinv <- link_parts$linkinv
  base_family$mu.eta <- link_parts$mu.eta
  base_family$valideta <- link_parts$valideta
  base_family$dev.resids <- .mlxs_binomial_dev_resids
  base_family$validmu <- .mlxs_binomial_validmu
  base_family$initialize_mlx <- function(
    y,
    weights,
    eta,
    mu,
    nobs,
    warn_noninteger = TRUE
  ) {
    if (any(y < 0 | y > 1)) {
      stop("y values must be 0 <= y <= 1", call. = FALSE)
    }
    m <- weights * y
    if (warn_noninteger && any(abs(m - round(m)) > 0.001)) {
      warning(
        "non-integer #successes in a binomial glm!",
        call. = FALSE
      )
    }
    mu <- (weights * y + 0.5) / (weights + 1)
    list(mu = mu, eta = base_family$linkfun(mu))
  }
  base_family
}

.mlxs_binomial_validmu <- function(mu) {
  all(is.finite(mu)) && all(mu > 0) && all(mu < 1)
}

.mlxs_binomial_dev_resids <- function(y, mu, wt) {
  if (length(y) == 0) {
    return(y)
  }

  y <- Rmlx::as_mlx(y)
  mu <- Rmlx::as_mlx(mu)
  wt <- Rmlx::as_mlx(wt)

  # Use Rmlx binary cross entropy with reduction="none"
  # Deviance = 2 * wt * (saturated_loglik - fitted_loglik)
  # where saturated_loglik = y*log(y) + (1-y)*log(1-y)
  # and fitted_loglik = y*log(mu) + (1-y)*log(1-mu)
  # BCE(mu, y) = -(y*log(mu) + (1-y)*log(1-mu)) = -fitted_loglik
  # So deviance = 2 * wt * (saturated_loglik + BCE)

  y_clamped <- .mlxs_binomial_clip_unit(y)
  saturated_loglik <- y * log(y_clamped) + (1 - y) * log(1 - y_clamped)

  bce <- Rmlx::mlx_binary_cross_entropy(mu, y, reduction = "none")
  2 * wt * (saturated_loglik + bce)
}

.mlxs_binomial_clip_unit <- function(x, eps = .mlxs_tail_epsilon(x)) {
  x <- Rmlx::as_mlx(x)
  x_dtype <- Rmlx::mlx_dtype(x)
  eps_scalar <- Rmlx::as_mlx(eps, dtype = x_dtype)
  upper_scalar <- Rmlx::as_mlx(1 - eps, dtype = x_dtype)

  x <- Rmlx::mlx_where(x < eps_scalar, eps_scalar, x)
  Rmlx::mlx_where(x > upper_scalar, upper_scalar, x)
}

.mlxs_binomial_link <- function(name) {
  switch(
    name,
    logit = .mlxs_logit_link(),
    log = .mlxs_log_link(),
    cloglog = .mlxs_cloglog_link(),
    cauchit = .mlxs_cauchit_link(),
    NULL
  )
}

.mlxs_logit_link <- function() {
  linkinv <- function(eta) {
    1 / (1 + exp(-eta))
  }
  mu_eta <- function(eta) {
    mu <- linkinv(eta)
    eps <- .mlxs_tail_epsilon(eta)
    Rmlx::mlx_maximum(mu * (1 - mu), eps)
  }
  list(
    linkfun = function(mu) {
      log(mu / (1 - mu))
    },
    linkinv = linkinv,
    mu.eta = mu_eta,
    valideta = function(eta) TRUE
  )
}

.mlxs_log_link <- function() {
  linkinv <- function(eta) {
    exp(eta)
  }
  list(
    linkfun = function(mu) {
      log(mu)
    },
    linkinv = linkinv,
    mu.eta = function(eta) {
      eps <- .mlxs_tail_epsilon(eta)
      deriv <- linkinv(eta)
      Rmlx::mlx_maximum(deriv, eps)
    },
    valideta = function(eta) all(is.finite(eta))
  )
}

.mlxs_cloglog_link <- function() {
  linkinv <- function(eta) {
    1 - exp(-exp(eta))
  }
  list(
    linkfun = function(mu) {
      log(-log(1 - mu))
    },
    linkinv = linkinv,
    mu.eta = function(eta) {
      eps <- .mlxs_tail_epsilon(eta)
      deriv <- exp(eta - exp(eta))
      Rmlx::mlx_maximum(deriv, eps)
    },
    valideta = function(eta) all(is.finite(eta))
  )
}

.mlxs_cauchit_link <- function() {
  linkinv <- function(eta) {
    atan(eta) / pi + 0.5
  }
  list(
    linkfun = function(mu) {
      tan(pi * (mu - 0.5))
    },
    linkinv = linkinv,
    mu.eta = function(eta) {
      eps <- .mlxs_tail_epsilon(eta)
      deriv <- 1 / (pi * (1 + eta^2))
      Rmlx::mlx_maximum(deriv, eps)
    },
    valideta = function(eta) all(is.finite(eta))
  )
}

.mlxs_identity_link <- function() {
  list(
    linkfun = function(mu) mu,
    linkinv = function(eta) eta,
    mu.eta = function(eta) {
      if (inherits(eta, "mlx")) {
        eta - eta + Rmlx::as_mlx(1)
      } else {
        rep.int(1, length(eta))
      }
    },
    valideta = function(eta) all(is.finite(eta))
  )
}

.mlxs_inverse_link <- function() {
  eta_adj <- function(eta) {
    eps <- .mlxs_tail_epsilon(eta)
    if (inherits(eta, "mlx")) {
      eta_sign <- Rmlx::mlx_where(eta >= 0, 1, -1)
      return(eta_sign * Rmlx::mlx_maximum(abs(eta), eps))
    }
    ifelse(eta >= 0, 1, -1) * pmax(abs(eta), eps)
  }

  list(
    linkfun = function(mu) 1 / mu,
    linkinv = function(eta) {
      1 / eta_adj(eta)
    },
    mu.eta = function(eta) {
      -1 / (eta_adj(eta)^2)
    },
    valideta = function(eta) all(is.finite(eta)) && all(eta != 0)
  )
}

.mlxs_sqrt_link <- function() {
  list(
    linkfun = function(mu) sqrt(mu),
    linkinv = function(eta) eta^2,
    mu.eta = function(eta) {
      if (inherits(eta, "mlx")) {
        2 * eta
      } else {
        2 * eta
      }
    },
    valideta = function(eta) all(is.finite(eta)) && all(eta > 0)
  )
}
