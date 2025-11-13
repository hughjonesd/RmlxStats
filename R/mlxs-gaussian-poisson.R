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
    # Use Rmlx MSE loss with reduction="none" for per-observation losses
    wt * Rmlx::mlx_mse_loss(mu, y, reduction = "none")
  }

  base_family$aic <- function(y, n, mu, wt, dev) {
    nobs <- length(y)
    nobs * (log(dev / nobs * 2 * pi) + 1) + 2 - sum(log(wt))
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

  base_family$dev.resids <- function(y, mu, wt) {
    eps <- 1e-6
    mu_clamped <- Rmlx::mlx_where(mu < eps, eps, mu)
    y_positive <- y > eps
    safe_y <- Rmlx::mlx_where(y_positive, y, eps)
    log_ratio <- log(safe_y / mu_clamped)
    term <- wt * (safe_y * log_ratio - (y - mu))
    base <- wt * mu
    res <- Rmlx::mlx_where(y_positive, term, base)
    2 * res
  }

  base_family$aic <- function(y, n, mu, wt, dev) {
    y_num <- as.numeric(as.matrix(y))
    mu_num <- as.numeric(as.matrix(mu))
    wt_num <- as.numeric(as.matrix(wt))
    -2 * sum(stats::dpois(y_num, mu_num, log = TRUE) * wt_num)
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
