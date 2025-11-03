#' MLX-friendly Gaussian family
#'
#' @inheritParams stats::gaussian
#' @return A family object compatible with `mlxs_glm()`.
#' @export
mlxs_gaussian <- function(link = "identity") {
  base_family <- stats::gaussian(link = link)

  link_parts <- switch(
    base_family$link,
    identity = .mlxs_identity_link(),
    log = .mlxs_log_link(),
    inverse = .mlxs_inverse_link(),
    stop("Unsupported link for mlxs_gaussian: ", base_family$link, call. = FALSE)
  )

  base_family$linkfun <- link_parts$linkfun
  base_family$linkinv <- link_parts$linkinv
  base_family$mu.eta <- link_parts$mu.eta
  base_family$valideta <- link_parts$valideta

  base_family$variance <- function(mu) {
    if (inherits(mu, "mlx")) {
      mu - mu + Rmlx::as_mlx(1)
    } else {
      rep.int(1, length(mu))
    }
  }

  base_family$dev.resids <- function(y, mu, wt) {
    if (inherits(y, "mlx") || inherits(mu, "mlx") || inherits(wt, "mlx")) {
      y_mlx <- Rmlx::as_mlx(y)
      mu_mlx <- Rmlx::as_mlx(mu)
      wt_mlx <- Rmlx::as_mlx(wt)
      diff <- y_mlx - mu_mlx
      wt_mlx * (diff * diff)
    } else {
      wt * ((y - mu)^2)
    }
  }

  base_family$aic <- function(y, n, mu, wt, dev) {
    y_num <- as.numeric(as.matrix(y))
    wt_num <- as.numeric(as.matrix(wt))
    nobs <- length(y_num)
    nobs * (log(dev / nobs * 2 * pi) + 1) + 2 - sum(log(wt_num))
  }

  base_family
}

#' MLX-friendly Poisson family
#'
#' @inheritParams stats::poisson
#' @return A family object compatible with `mlxs_glm()`.
#' @export
mlxs_poisson <- function(link = "log") {
  base_family <- stats::poisson(link = link)

  link_parts <- switch(
    base_family$link,
    log = .mlxs_log_link(),
    identity = .mlxs_identity_link(),
    sqrt = .mlxs_sqrt_link(),
    stop("Unsupported link for mlxs_poisson: ", base_family$link, call. = FALSE)
  )

  base_family$linkfun <- link_parts$linkfun
  base_family$linkinv <- link_parts$linkinv
  base_family$mu.eta <- link_parts$mu.eta
  base_family$valideta <- link_parts$valideta

  base_family$variance <- function(mu) {
    if (inherits(mu, "mlx")) {
      mu
    } else {
      mu
    }
  }

  base_family$dev.resids <- function(y, mu, wt) {
    if (inherits(y, "mlx") || inherits(mu, "mlx") || inherits(wt, "mlx")) {
      y_mlx <- Rmlx::as_mlx(y)
      mu_mlx <- Rmlx::as_mlx(mu)
      wt_mlx <- Rmlx::as_mlx(wt)

      eps <- Rmlx::as_mlx(1e-6)
      mu_clamped <- Rmlx::mlx_where(mu_mlx < eps, eps, mu_mlx)
      y_positive <- y_mlx > eps
      safe_y <- Rmlx::mlx_where(y_positive, y_mlx, eps)
      log_ratio <- log(safe_y / mu_clamped)
      term <- wt_mlx * (safe_y * log_ratio - (y_mlx - mu_mlx))
      base <- wt_mlx * mu_mlx
      res <- Rmlx::mlx_where(y_positive, term, base)
      2 * res
    } else {
      r <- mu * wt
      p <- which(y > 0)
      r[p] <- (wt * (y * log(y/mu) - (y - mu)))[p]
      2 * r
    }
  }

  base_family$aic <- function(y, n, mu, wt, dev) {
    y_num <- as.numeric(as.matrix(y))
    mu_num <- as.numeric(as.matrix(mu))
    wt_num <- as.numeric(as.matrix(wt))
    -2 * sum(dpois(y_num, mu_num, log = TRUE) * wt_num)
  }

  base_family$validmu <- function(mu) {
    mu_num <- as.numeric(as.matrix(mu))
    all(is.finite(mu_num)) && all(mu_num > 0)
  }

  base_family
}

#' MLX-friendly quasibinomial family
#'
#' @inheritParams stats::quasibinomial
#' @return A family object compatible with `mlxs_glm()`.
#' @export
mlxs_quasibinomial <- function(link = "logit") {
  fam <- mlxs_binomial(link)
  fam$family <- "quasibinomial"
  fam$aic <- function(...) NA_real_
  fam$dispersion <- NA_real_
  fam
}

#' MLX-friendly quasipoisson family
#'
#' @inheritParams stats::quasipoisson
#' @return A family object compatible with `mlxs_glm()`.
#' @export
mlxs_quasipoisson <- function(link = "log") {
  fam <- mlxs_poisson(link)
  fam$family <- "quasipoisson"
  fam$aic <- function(...) NA_real_
  fam$dispersion <- NA_real_
  fam
}
