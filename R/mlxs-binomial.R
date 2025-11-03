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

  eps <- 1e-6
  mu_clamped <- .mlxs_binomial_clip_unit(mu, eps)
  y_clamped <- .mlxs_binomial_clip_unit(y, eps)

  term1 <- y * (log(y_clamped) - log(mu_clamped))
  term2 <- (1 - y) * (log(1 - y_clamped) - log(1 - mu_clamped))
  2 * wt * (term1 + term2)
}

.mlxs_binomial_clip_unit <- function(x, eps) {
  x <- Rmlx::as_mlx(x)
  eps_scalar <- Rmlx::as_mlx(eps)
  upper_scalar <- Rmlx::as_mlx(1 - eps)

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
    eps <- Rmlx::as_mlx(1e-6)
    Rmlx::mlx_where(mu * (1 - mu) < eps, eps, mu * (1 - mu))
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
      eps <- Rmlx::as_mlx(1e-6)
      deriv <- linkinv(eta)
      Rmlx::mlx_where(deriv < eps, eps, deriv)
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
      eps <- Rmlx::as_mlx(1e-6)
      deriv <- exp(eta - exp(eta))
      Rmlx::mlx_where(deriv < eps, eps, deriv)
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
      eps <- Rmlx::as_mlx(1e-6)
      deriv <- 1 / (pi * (1 + eta^2))
      Rmlx::mlx_where(deriv < eps, eps, deriv)
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
  list(
    linkfun = function(mu) 1 / mu,
    linkinv = function(eta) {
      eps <- Rmlx::as_mlx(1e-6)
      eta_adj <- if (inherits(eta, "mlx")) {
        Rmlx::mlx_where(abs(eta) < eps, Rmlx::mlx_where(eta >= 0, eps, -eps), eta)
      } else {
        pmax(pmin(eta, -1e-6), 1e-6)
      }
      1 / eta_adj
    },
    mu.eta = function(eta) {
      if (inherits(eta, "mlx")) {
        eta_adj <- Rmlx::mlx_where(abs(eta) < Rmlx::as_mlx(1e-6),
                                   Rmlx::mlx_where(eta >= 0, Rmlx::as_mlx(1e-6), Rmlx::as_mlx(-1e-6)),
                                   eta)
        -1 / (eta_adj^2)
      } else {
        -1 / (pmax(pmin(eta, -1e-6), 1e-6)^2)
      }
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
