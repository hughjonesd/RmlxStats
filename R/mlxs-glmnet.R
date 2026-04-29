#' MLX-backed elastic net regression
#'
#' Fit lasso or elastic-net penalised regression paths using MLX arrays for the
#' heavy linear algebra. Dense Gaussian and binomial paths stay on the MLX
#' backend throughout the iterative updates, with repeated chunk updates traced
#' through [Rmlx::mlx_compile()] to reduce host overhead.
#'
#' @note `glmnet::glmnet()` is faster on smaller problems. Very roughly
#'  as of April 2026, `mlxs_glmnet()` gets competitive at n x p = 5,000,000 
#'  or greater.
#'
#' @param x Numeric matrix of predictors (observations in rows).
#' @param y Numeric response vector (for binomial, values must be 0/1).
#' @param family MLX-aware family object, e.g. [mlxs_gaussian()] or
#'   [mlxs_binomial()].
#' @param alpha Elastic-net mixing parameter (1 = lasso, currently alpha must
#'   be strictly positive).
#' @param lambda Optional decreasing sequence of penalty values. If `NULL`,
#'   a sequence of length `nlambda` is generated from `lambda_max` down to
#'   `lambda_max * lambda_min_ratio`.
#' @param nlambda Length of automatically generated lambda path.
#' @param lambda_min_ratio Smallest lambda as a fraction of `lambda_max`.
#' @param standardize Should columns of `x` be scaled before fitting?
#' @param intercept Should an intercept be fit?
#' @param use_strong_rules Retained for API compatibility. The dense MLX solver
#'   keeps all coefficients on device, so this flag currently does not change
#'   the computation.
#' @param maxit Maximum proximal-gradient iterations per lambda value.
#' @param tol Convergence tolerance on the coefficient updates.
#' @return An object of class `mlxs_glmnet` containing the fitted coefficient
#'   path, intercepts, lambda sequence, and scaling information.
#' @export
mlxs_glmnet <- function(x,
                        y,
                        family = mlxs_gaussian(),
                        alpha = 1,
                        lambda = NULL,
                        nlambda = 100,
                        lambda_min_ratio = 1e-4,
                        standardize = TRUE,
                        intercept = TRUE,
                        use_strong_rules = TRUE,
                        maxit = 1000,
                        tol = 1e-6) {
  family_name <- family$family
  if (!family_name %in% c("gaussian", "binomial", "quasibinomial")) {
    stop("mlxs_glmnet() currently supports gaussian and binomial families.",
         call. = FALSE)
  }
  if (alpha <= 0) {
    stop("alpha must be > 0 for the current MLX elastic-net implementation.",
         call. = FALSE)
  }

  x <- as.matrix(x)
  y <- as.numeric(y)
  if (nrow(x) != length(y)) {
    stop("x and y must have the same number of observations.", call. = FALSE)
  }

  n_obs <- nrow(x)
  n_pred <- ncol(x)
  chunk_size <- 8L

  design <- .mlxs_glmnet_prepare_design(x, standardize = standardize,
                                        intercept = intercept)
  x_mlx <- design$x
  x_center <- design$x_center
  x_scale <- design$x_scale

  if (family_name == "gaussian") {
    fit <- .mlxs_glmnet_fit_gaussian(
      x_mlx = x_mlx,
      y = y,
      alpha = alpha,
      lambda = lambda,
      nlambda = nlambda,
      lambda_min_ratio = lambda_min_ratio,
      intercept = intercept,
      maxit = maxit,
      tol = tol,
      chunk_size = chunk_size
    )
  } else {
    fit <- .mlxs_glmnet_fit_binomial(
      x_mlx = x_mlx,
      y = y,
      alpha = alpha,
      lambda = lambda,
      nlambda = nlambda,
      lambda_min_ratio = lambda_min_ratio,
      intercept = intercept,
      maxit = maxit,
      tol = tol,
      chunk_size = chunk_size
    )
  }

  n_lambda <- length(fit$lambda)
  beta_store_mlx <- fit$beta
  intercept_store_mlx <- fit$intercept

  beta_unscaled_mlx <- beta_store_mlx / x_scale
  if (intercept) {
    intercept_adjustment <- Rmlx::colSums(beta_unscaled_mlx * x_center)
    intercept_adjustment <- Rmlx::mlx_reshape(intercept_adjustment,
                                              c(n_lambda, 1L))
    intercept_unscaled_mlx <- intercept_store_mlx - intercept_adjustment
  } else {
    intercept_unscaled_mlx <- intercept_store_mlx
  }

  beta_unscaled <- as.matrix(beta_unscaled_mlx)
  intercept_unscaled <- as.numeric(intercept_unscaled_mlx)

  rownames(beta_unscaled) <- colnames(x)
  result <- list(
    a0 = intercept_unscaled,
    beta = beta_unscaled,
    lambda = fit$lambda,
    alpha = alpha,
    family = family_name,
    standardize = standardize,
    intercept = intercept,
    use_strong_rules = use_strong_rules,
    x_center = x_center,
    x_scale = x_scale,
    call = match.call()
  )

  class(result) <- "mlxs_glmnet"
  result
}

.mlxs_glmnet_prepare_design <- function(x,
                                        standardize,
                                        intercept) {
  x_mlx <- Rmlx::as_mlx(x)
  n_pred <- ncol(x)

  if (standardize || intercept) {
    x_proc <- scale(x_mlx, center = intercept, scale = standardize)
    x_center <- attr(x_proc, "scaled:center")
    x_scale <- attr(x_proc, "scaled:scale")
  } else {
    x_proc <- x_mlx
    x_center <- NULL
    x_scale <- NULL
  }

  if (is.null(x_center)) {
    x_center <- Rmlx::mlx_zeros(c(1L, n_pred))
  }
  if (is.null(x_scale)) {
    x_scale <- Rmlx::mlx_ones(c(1L, n_pred))
  }

  list(
    x = x_proc,
    x_center = Rmlx::mlx_reshape(x_center, c(n_pred, 1L)),
    x_scale = Rmlx::mlx_reshape(x_scale, c(n_pred, 1L))
  )
}

.mlxs_glmnet_fit_gaussian <- function(x_mlx,
                                      y,
                                      alpha,
                                      lambda,
                                      nlambda,
                                      lambda_min_ratio,
                                      intercept,
                                      maxit,
                                      tol,
                                      chunk_size) {
  n_obs <- nrow(x_mlx)
  n_pred <- ncol(x_mlx)
  n_lambda <- if (is.null(lambda)) nlambda else length(lambda)

  solver <- .mlxs_glmnet_choose_gaussian_solver(
    n_obs = n_obs,
    n_pred = n_pred,
    n_lambda = n_lambda
  )

  if (identical(solver, "gram")) {
    return(.mlxs_glmnet_fit_gaussian_gram(
      x_mlx = x_mlx,
      y = y,
      alpha = alpha,
      lambda = lambda,
      nlambda = nlambda,
      lambda_min_ratio = lambda_min_ratio,
      intercept = intercept,
      maxit = maxit,
      tol = tol,
      chunk_size = chunk_size
    ))
  }

  .mlxs_glmnet_fit_gaussian_dense(
    x_mlx = x_mlx,
    y = y,
    alpha = alpha,
    lambda = lambda,
    nlambda = nlambda,
    lambda_min_ratio = lambda_min_ratio,
    intercept = intercept,
    maxit = maxit,
    tol = tol,
    chunk_size = chunk_size
  )
}

.mlxs_glmnet_fit_gaussian_dense <- function(x_mlx,
                                            y,
                                            alpha,
                                            lambda,
                                            nlambda,
                                            lambda_min_ratio,
                                            intercept,
                                            maxit,
                                            tol,
                                            chunk_size) {
  n_obs <- nrow(x_mlx)
  n_pred <- ncol(x_mlx)
  y_mean <- if (intercept) mean(y) else 0
  y_mlx <- Rmlx::mlx_reshape(Rmlx::as_mlx(y - y_mean), c(n_obs, 1L))
  shape_sig <- paste(n_obs, n_pred, sep = "x")

  lambda_max <- .mlxs_glmnet_lambda_max(x_mlx, -y_mlx, n_obs, alpha)
  lambda <- .mlxs_glmnet_lambda_path(lambda, lambda_max, nlambda,
                                     lambda_min_ratio)
  n_lambda <- length(lambda)

  beta_mlx <- Rmlx::mlx_zeros(c(n_pred, 1L))
  eta_mlx <- Rmlx::mlx_zeros(c(n_obs, 1L))
  residual_mlx <- -y_mlx
  beta_store_mlx <- Rmlx::mlx_zeros(c(n_pred, n_lambda))
  intercept_store_mlx <- Rmlx::mlx_zeros(c(n_lambda, 1L))
  intercept_mlx <- Rmlx::mlx_matrix(y_mean, nrow = 1L, ncol = 1L)

  gram_mlx <- crossprod(x_mlx) / n_obs
  base_lipschitz <- as.numeric(max(Rmlx::colSums(abs(gram_mlx))))
  n_obs_mlx <- Rmlx::as_mlx(n_obs)
  zero_mlx <- Rmlx::as_mlx(0)

  for (idx in seq_along(lambda)) {
    lambda_val <- lambda[idx]
    step <- 1 / (base_lipschitz + lambda_val * (1 - alpha))
    thresh <- lambda_val * alpha * step
    ridge_penalty <- lambda_val * (1 - alpha)
    step_mlx <- Rmlx::as_mlx(step)
    thresh_mlx <- Rmlx::as_mlx(thresh)
    ridge_penalty_mlx <- Rmlx::as_mlx(ridge_penalty)

    remaining <- maxit
    while (remaining > 0L) {
      n_steps <- min(chunk_size, remaining)
      state <- .mlxs_glmnet_get_compiled_chunk(
        "gaussian",
        n_steps,
        shape_sig = shape_sig
      )(
        x_mlx,
        beta_mlx,
        eta_mlx,
        residual_mlx,
        y_mlx,
        n_obs_mlx,
        step_mlx,
        thresh_mlx,
        ridge_penalty_mlx,
        zero_mlx
      )

      beta_mlx <- state$beta
      eta_mlx <- state$eta
      residual_mlx <- state$residual
      remaining <- remaining - n_steps

      if (as.logical(state$delta_max < tol)) {
        break
      }
    }

    beta_store_mlx <- Rmlx::mlx_slice_update(
      beta_store_mlx,
      beta_mlx,
      start = c(1L, idx),
      stop = c(n_pred, idx)
    )
    intercept_store_mlx <- Rmlx::mlx_slice_update(
      intercept_store_mlx,
      intercept_mlx,
      start = c(idx, 1L),
      stop = c(idx, 1L)
    )
  }

  list(
    beta = beta_store_mlx,
    intercept = intercept_store_mlx,
    lambda = lambda
  )
}

.mlxs_glmnet_fit_gaussian_gram <- function(x_mlx,
                                           y,
                                           alpha,
                                           lambda,
                                           nlambda,
                                           lambda_min_ratio,
                                           intercept,
                                           maxit,
                                           tol,
                                           chunk_size) {
  n_obs <- nrow(x_mlx)
  n_pred <- ncol(x_mlx)
  y_mean <- if (intercept) mean(y) else 0
  y_mlx <- Rmlx::mlx_reshape(Rmlx::as_mlx(y - y_mean), c(n_obs, 1L))

  gram_mlx <- crossprod(x_mlx) / n_obs
  xy_mlx <- crossprod(x_mlx, y_mlx) / n_obs
  shape_sig <- paste(n_pred, n_pred, sep = "x")

  lambda_max <- max(abs(as.numeric(xy_mlx))) / max(alpha, 1e-8)
  if (is.finite(lambda_max) && lambda_max == 0) {
    lambda_max <- 1
  }
  lambda <- .mlxs_glmnet_lambda_path(lambda, lambda_max, nlambda,
                                     lambda_min_ratio)
  n_lambda <- length(lambda)

  beta_mlx <- Rmlx::mlx_zeros(c(n_pred, 1L))
  z_mlx <- beta_mlx
  t_prev <- 1
  beta_store_mlx <- Rmlx::mlx_zeros(c(n_pred, n_lambda))
  intercept_store_mlx <- Rmlx::mlx_zeros(c(n_lambda, 1L))
  intercept_mlx <- Rmlx::mlx_matrix(y_mean, nrow = 1L, ncol = 1L)
  gram_lipschitz <- as.numeric(max(Rmlx::colSums(abs(gram_mlx))))
  effective_maxit <- min(maxit, 200L)
  zero_mlx <- Rmlx::as_mlx(0)
  one_mlx <- Rmlx::as_mlx(1)
  four_mlx <- Rmlx::as_mlx(4)

  for (idx in seq_along(lambda)) {
    lambda_val <- lambda[idx]
    ridge_penalty <- lambda_val * (1 - alpha)
    step <- 1 / (gram_lipschitz + ridge_penalty)
    thresh <- lambda_val * alpha * step
    step_mlx <- Rmlx::as_mlx(step)
    thresh_mlx <- Rmlx::as_mlx(thresh)
    ridge_penalty_mlx <- Rmlx::as_mlx(ridge_penalty)

    remaining <- effective_maxit
    z_mlx <- beta_mlx
    t_prev_mlx <- one_mlx

    while (remaining > 0L) {
      n_steps <- min(chunk_size, remaining)
      state <- .mlxs_glmnet_get_compiled_chunk(
        "gaussian_gram",
        n_steps,
        shape_sig = shape_sig
      )(
        gram_mlx,
        xy_mlx,
        beta_mlx,
        z_mlx,
        t_prev_mlx,
        step_mlx,
        thresh_mlx,
        ridge_penalty_mlx,
        zero_mlx,
        one_mlx,
        four_mlx
      )

      beta_mlx <- state$beta
      z_mlx <- state$z
      t_prev_mlx <- state$t_prev
      remaining <- remaining - n_steps

      if (as.logical(state$delta_max < tol)) {
        break
      }
    }

    beta_store_mlx <- Rmlx::mlx_slice_update(
      beta_store_mlx,
      beta_mlx,
      start = c(1L, idx),
      stop = c(n_pred, idx)
    )
    intercept_store_mlx <- Rmlx::mlx_slice_update(
      intercept_store_mlx,
      intercept_mlx,
      start = c(idx, 1L),
      stop = c(idx, 1L)
    )
  }

  list(
    beta = beta_store_mlx,
    intercept = intercept_store_mlx,
    lambda = lambda
  )
}

.mlxs_glmnet_fit_binomial <- function(x_mlx,
                                      y,
                                      alpha,
                                      lambda,
                                      nlambda,
                                      lambda_min_ratio,
                                      intercept,
                                      maxit,
                                      tol,
                                      chunk_size) {
  if (!all(y %in% c(0, 1))) {
    stop("Binomial family requires a 0/1 response.", call. = FALSE)
  }

  n_obs <- nrow(x_mlx)
  n_pred <- ncol(x_mlx)
  y_mlx <- Rmlx::mlx_reshape(Rmlx::as_mlx(y), c(n_obs, 1L))
  ones_mlx <- Rmlx::mlx_ones(c(n_obs, 1L))
  shape_sig <- paste(n_obs, n_pred, fit_intercept = intercept, sep = "x")

  p_hat <- mean(y)
  p_hat <- min(max(p_hat, 1e-6), 1 - 1e-6)
  intercept_val <- if (intercept) log(p_hat / (1 - p_hat)) else 0
  intercept_mlx <- Rmlx::mlx_matrix(intercept_val, nrow = 1L, ncol = 1L)
  beta_mlx <- Rmlx::mlx_zeros(c(n_pred, 1L))
  eta_mlx <- if (intercept) ones_mlx * intercept_mlx else {
    Rmlx::mlx_zeros(c(n_obs, 1L))
  }
  residual_mlx <- (1 / (1 + exp(-eta_mlx))) - y_mlx

  lambda_max <- .mlxs_glmnet_lambda_max(x_mlx, residual_mlx, n_obs, alpha)
  lambda <- .mlxs_glmnet_lambda_path(lambda, lambda_max, nlambda,
                                     lambda_min_ratio)
  n_lambda <- length(lambda)

  beta_store_mlx <- Rmlx::mlx_zeros(c(n_pred, n_lambda))
  intercept_store_mlx <- Rmlx::mlx_zeros(c(n_lambda, 1L))
  col_sq_sums <- as.numeric(Rmlx::colSums(x_mlx^2))
  base_lipschitz <- 0.25 * max(col_sq_sums) / n_obs
  n_obs_mlx <- Rmlx::as_mlx(n_obs)
  zero_mlx <- Rmlx::as_mlx(0)

  for (idx in seq_along(lambda)) {
    lambda_val <- lambda[idx]
    step <- 1 / (base_lipschitz + lambda_val * (1 - alpha))
    thresh <- lambda_val * alpha * step
    ridge_penalty <- lambda_val * (1 - alpha)
    step_mlx <- Rmlx::as_mlx(step)
    thresh_mlx <- Rmlx::as_mlx(thresh)
    ridge_penalty_mlx <- Rmlx::as_mlx(ridge_penalty)

    remaining <- maxit
    while (remaining > 0L) {
      n_steps <- min(chunk_size, remaining)
      state <- .mlxs_glmnet_get_compiled_chunk(
        "binomial",
        n_steps,
        fit_intercept = intercept,
        shape_sig = shape_sig
      )(
        x_mlx,
        beta_mlx,
        intercept_mlx,
        eta_mlx,
        residual_mlx,
        y_mlx,
        ones_mlx,
        n_obs_mlx,
        step_mlx,
        thresh_mlx,
        ridge_penalty_mlx,
        zero_mlx
      )

      beta_mlx <- state$beta
      intercept_mlx <- state$intercept
      eta_mlx <- state$eta
      residual_mlx <- state$residual
      remaining <- remaining - n_steps

      if (as.logical(state$delta_max < tol) &&
          as.logical(state$intercept_delta_max < tol)) {
        break
      }
    }

    beta_store_mlx <- Rmlx::mlx_slice_update(
      beta_store_mlx,
      beta_mlx,
      start = c(1L, idx),
      stop = c(n_pred, idx)
    )
    intercept_store_mlx <- Rmlx::mlx_slice_update(
      intercept_store_mlx,
      intercept_mlx,
      start = c(idx, 1L),
      stop = c(idx, 1L)
    )
  }

  list(
    beta = beta_store_mlx,
    intercept = intercept_store_mlx,
    lambda = lambda
  )
}

.mlxs_glmnet_lambda_max <- function(x_mlx, residual_mlx, n_obs, alpha) {
  z0_mlx <- crossprod(x_mlx, residual_mlx) / n_obs
  lambda_max <- max(abs(as.numeric(z0_mlx))) / max(alpha, 1e-8)
  if (is.finite(lambda_max) && lambda_max == 0) {
    lambda_max <- 1
  }
  lambda_max
}

.mlxs_glmnet_lambda_path <- function(lambda,
                                     lambda_max,
                                     nlambda,
                                     lambda_min_ratio) {
  if (is.null(lambda)) {
    lambda_min <- lambda_max * lambda_min_ratio
    exp(seq(log(lambda_max), log(lambda_min), length.out = nlambda))
  } else {
    sort(as.numeric(lambda), decreasing = TRUE)
  }
}

.mlxs_glmnet_choose_gaussian_solver <- function(n_obs, n_pred, n_lambda) {
  if (n_lambda >= 10L && n_pred <= 1024L && n_obs >= 50L * n_pred) {
    "gram"
  } else {
    "dense"
  }
}

.mlxs_soft_threshold <- function(z, thresh) {
  sign(z) * Rmlx::mlx_maximum(abs(z) - thresh, 0)
}
