#' Run uncompiled Gaussian elastic-net updates
#'
#' Reference implementation for a fixed number of dense Gaussian
#' proximal-gradient steps. The compiled factories below mirror this state
#' transition and are used in normal path fitting.
#'
#' @param x_mlx MLX predictor matrix.
#' @param beta_mlx Current coefficient column vector.
#' @param eta_mlx Current linear predictor.
#' @param residual_mlx Current residual column vector.
#' @param y_mlx Centered response column vector.
#' @param n_obs Number of observations.
#' @param step Gradient step size.
#' @param thresh Soft-thresholding value.
#' @param ridge_penalty Ridge part of the elastic-net penalty.
#' @param n_steps Number of update steps to run.
#' @return List with updated `beta`, `eta`, `residual`, and `delta_max`.
#' @noRd
.mlxs_glmnet_gaussian_chunk <- function(x_mlx,
                                        beta_mlx,
                                        eta_mlx,
                                        residual_mlx,
                                        y_mlx,
                                        n_obs,
                                        step,
                                        thresh,
                                        ridge_penalty,
                                        n_steps) {
  delta_max <- Rmlx::mlx_scalar(0)

  for (i in seq_len(n_steps)) {
    grad_mlx <- crossprod(x_mlx, residual_mlx) / n_obs
    if (ridge_penalty != 0) {
      grad_mlx <- grad_mlx + beta_mlx * ridge_penalty
    }

    beta_new_mlx <- .mlxs_soft_threshold(beta_mlx - step * grad_mlx, thresh)
    delta_mlx <- beta_new_mlx - beta_mlx

    eta_mlx <- eta_mlx + x_mlx %*% delta_mlx
    residual_mlx <- eta_mlx - y_mlx
    beta_mlx <- beta_new_mlx

    delta_max <- Rmlx::mlx_maximum(delta_max, max(abs(delta_mlx)))
  }

  list(
    beta = beta_mlx,
    eta = eta_mlx,
    residual = residual_mlx,
    delta_max = delta_max
  )
}

#' Run uncompiled Gram Gaussian elastic-net updates
#'
#' Reference implementation for the tall-design Gaussian solver. It updates the
#' FISTA state using precomputed `X'X` and `X'y` matrices.
#'
#' @param gram_mlx MLX Gram matrix.
#' @param xy_mlx MLX crossproduct of predictors and response.
#' @param beta_mlx Current coefficient column vector.
#' @param z_mlx Current accelerated coefficient state.
#' @param t_prev Current FISTA momentum scalar.
#' @param step Gradient step size.
#' @param thresh Soft-thresholding value.
#' @param ridge_penalty Ridge part of the elastic-net penalty.
#' @param n_steps Number of update steps to run.
#' @return List with updated `beta`, `z`, `t_prev`, and `delta_max`.
#' @noRd
.mlxs_glmnet_gaussian_gram_chunk <- function(gram_mlx,
                                             xy_mlx,
                                             beta_mlx,
                                             z_mlx,
                                             t_prev,
                                             step,
                                             thresh,
                                             ridge_penalty,
                                             n_steps) {
  delta_max <- Rmlx::mlx_scalar(0)

  for (i in seq_len(n_steps)) {
    grad_mlx <- gram_mlx %*% z_mlx - xy_mlx
    if (ridge_penalty != 0) {
      grad_mlx <- grad_mlx + z_mlx * ridge_penalty
    }

    beta_new_mlx <- .mlxs_soft_threshold(z_mlx - step * grad_mlx, thresh)
    delta_mlx <- beta_new_mlx - beta_mlx
    delta_max <- Rmlx::mlx_maximum(delta_max, max(abs(delta_mlx)))

    t_next <- (1 + sqrt(1 + 4 * t_prev^2)) / 2
    z_mlx <- beta_new_mlx + ((t_prev - 1) / t_next) * delta_mlx
    beta_mlx <- beta_new_mlx
    t_prev <- t_next
  }

  list(
    beta = beta_mlx,
    z = z_mlx,
    t_prev = t_prev,
    delta_max = delta_max
  )
}

#' Run uncompiled binomial elastic-net updates
#'
#' Reference implementation for a fixed number of logistic proximal-gradient
#' steps. It updates coefficients, optional intercept, linear predictor, and
#' logistic residuals for one chunk of the lambda-path solver.
#'
#' @param x_mlx MLX predictor matrix.
#' @param beta_mlx Current coefficient column vector.
#' @param intercept_mlx Current intercept scalar.
#' @param eta_mlx Current linear predictor.
#' @param residual_mlx Current `mu - y` residual column vector.
#' @param y_mlx Binary response column vector.
#' @param ones_mlx Column vector of ones for intercept updates.
#' @param n_obs Number of observations.
#' @param step Gradient step size.
#' @param thresh Soft-thresholding value.
#' @param ridge_penalty Ridge part of the elastic-net penalty.
#' @param n_steps Number of update steps to run.
#' @param fit_intercept Logical; update the intercept.
#' @return List with updated path state and convergence deltas.
#' @noRd
.mlxs_glmnet_binomial_chunk <- function(x_mlx,
                                        beta_mlx,
                                        intercept_mlx,
                                        eta_mlx,
                                        residual_mlx,
                                        y_mlx,
                                        ones_mlx,
                                        n_obs,
                                        step,
                                        thresh,
                                        ridge_penalty,
                                        n_steps,
                                        fit_intercept) {
  delta_max <- Rmlx::mlx_scalar(0)
  intercept_delta_max <- Rmlx::mlx_scalar(0)

  for (i in seq_len(n_steps)) {
    grad_mlx <- crossprod(x_mlx, residual_mlx) / n_obs
    if (ridge_penalty != 0) {
      grad_mlx <- grad_mlx + beta_mlx * ridge_penalty
    }

    beta_new_mlx <- .mlxs_soft_threshold(beta_mlx - step * grad_mlx, thresh)
    delta_mlx <- beta_new_mlx - beta_mlx

    if (fit_intercept) {
      intercept_delta_mlx <- step * (Rmlx::mlx_sum(residual_mlx) / n_obs)
      intercept_mlx <- intercept_mlx - intercept_delta_mlx
      eta_mlx <- eta_mlx + x_mlx %*% delta_mlx - ones_mlx * intercept_delta_mlx
      intercept_delta_max <- Rmlx::mlx_maximum(
        intercept_delta_max,
        abs(intercept_delta_mlx)
      )
    } else {
      eta_mlx <- eta_mlx + x_mlx %*% delta_mlx
    }

    mu_mlx <- 1 / (1 + exp(-eta_mlx))
    residual_mlx <- mu_mlx - y_mlx
    beta_mlx <- beta_new_mlx

    delta_max <- Rmlx::mlx_maximum(delta_max, max(abs(delta_mlx)))
  }

  list(
    beta = beta_mlx,
    intercept = intercept_mlx,
    eta = eta_mlx,
    residual = residual_mlx,
    delta_max = delta_max,
    intercept_delta_max = intercept_delta_max
  )
}

.mlxs_glmnet_chunk_cache <- new.env(parent = emptyenv())

#' Clear cached compiled glmnet chunks
#'
#' Used before cross-validation to avoid retaining compiled functions across
#' repeated fold fits with different shapes.
#'
#' @return Invisibly removes all objects from the chunk cache environment.
#' @noRd
.mlxs_glmnet_clear_chunk_cache <- function() {
  rm(
    list = ls(envir = .mlxs_glmnet_chunk_cache, all.names = TRUE),
    envir = .mlxs_glmnet_chunk_cache
  )
}

#' Build a cache key for compiled glmnet chunks
#'
#' Encodes the solver kind, chunk length, intercept mode, and shape signature so
#' repeated calls can reuse compatible compiled update functions.
#'
#' @param kind Solver kind: `"gaussian"`, `"gaussian_gram"`, or `"binomial"`.
#' @param n_steps Number of update steps inside the compiled function.
#' @param fit_intercept Optional intercept flag for binomial chunks.
#' @param shape_sig Optional string describing the relevant array shape.
#' @return Character cache key.
#' @noRd
.mlxs_glmnet_chunk_key <- function(kind,
                                   n_steps,
                                   fit_intercept = NULL,
                                   shape_sig = NULL) {
  shape_part <- shape_sig %||% ""
  if (is.null(fit_intercept)) {
    paste(kind, n_steps, shape_part, sep = "::")
  } else {
    paste(kind, n_steps, fit_intercept, shape_part, sep = "::")
  }
}

#' Retrieve or compile a glmnet update chunk
#'
#' Central cache accessor used by all glmnet path solvers. It compiles the
#' requested fixed-length update function on first use and returns the cached
#' compiled function thereafter.
#'
#' @inheritParams .mlxs_glmnet_chunk_key
#' @return Compiled R function that advances one solver state by `n_steps`.
#' @noRd
.mlxs_glmnet_get_compiled_chunk <- function(kind,
                                            n_steps,
                                            fit_intercept = NULL,
                                            shape_sig = NULL) {
  key <- .mlxs_glmnet_chunk_key(
    kind,
    n_steps,
    fit_intercept,
    shape_sig
  )
  if (!exists(key, envir = .mlxs_glmnet_chunk_cache, inherits = FALSE)) {
    chunk_fn <- switch(
      kind,
      gaussian = .mlxs_glmnet_make_gaussian_compiled_chunk(n_steps),
      gaussian_gram = .mlxs_glmnet_make_gaussian_gram_compiled_chunk(n_steps),
      binomial = .mlxs_glmnet_make_binomial_compiled_chunk(
        n_steps = n_steps,
        fit_intercept = fit_intercept
      ),
      stop("Unknown chunk kind: ", kind, call. = FALSE)
    )
    assign(key, chunk_fn, envir = .mlxs_glmnet_chunk_cache)
  }
  get(key, envir = .mlxs_glmnet_chunk_cache, inherits = FALSE)
}

#' Compile dense Gaussian glmnet updates
#'
#' Factory for the dense Gaussian solver. The returned function captures
#' `n_steps` so Rmlx can trace a fixed loop body for repeated lambda-path
#' iterations.
#'
#' @param n_steps Number of proximal-gradient steps in the compiled chunk.
#' @return Compiled function that updates dense Gaussian path state.
#' @noRd
.mlxs_glmnet_make_gaussian_compiled_chunk <- function(n_steps) {
  Rmlx::mlx_compile(function(x_mlx,
                             beta_mlx,
                             eta_mlx,
                             residual_mlx,
                             y_mlx,
                             n_obs_mlx,
                             step_mlx,
                             thresh_mlx,
                             ridge_penalty_mlx,
                             zero_mlx) {
    delta_max <- zero_mlx

    for (i in seq_len(n_steps)) {
      grad_mlx <- crossprod(x_mlx, residual_mlx) / n_obs_mlx
      grad_mlx <- grad_mlx + beta_mlx * ridge_penalty_mlx

      beta_new_mlx <- .mlxs_soft_threshold(
        beta_mlx - step_mlx * grad_mlx,
        thresh_mlx
      )
      delta_mlx <- beta_new_mlx - beta_mlx

      eta_mlx <- eta_mlx + x_mlx %*% delta_mlx
      residual_mlx <- eta_mlx - y_mlx
      beta_mlx <- beta_new_mlx

      delta_max <- Rmlx::mlx_maximum(delta_max, max(abs(delta_mlx)))
    }

    list(
      beta = beta_mlx,
      eta = eta_mlx,
      residual = residual_mlx,
      delta_max = delta_max
    )
  })
}

#' Compile Gram Gaussian glmnet updates
#'
#' Factory for the tall-design Gaussian solver. The returned function advances
#' the FISTA state using precomputed Gram data.
#'
#' @param n_steps Number of proximal-gradient steps in the compiled chunk.
#' @return Compiled function that updates Gram-solver path state.
#' @noRd
.mlxs_glmnet_make_gaussian_gram_compiled_chunk <- function(n_steps) {
  Rmlx::mlx_compile(function(gram_mlx,
                             xy_mlx,
                             beta_mlx,
                             z_mlx,
                             t_prev_mlx,
                             step_mlx,
                             thresh_mlx,
                             ridge_penalty_mlx,
                             zero_mlx,
                             one_mlx,
                             four_mlx) {
    delta_max <- zero_mlx
    t_prev_local <- t_prev_mlx

    for (i in seq_len(n_steps)) {
      grad_mlx <- gram_mlx %*% z_mlx - xy_mlx
      grad_mlx <- grad_mlx + z_mlx * ridge_penalty_mlx

      beta_new_mlx <- .mlxs_soft_threshold(
        z_mlx - step_mlx * grad_mlx,
        thresh_mlx
      )
      delta_mlx <- beta_new_mlx - beta_mlx
      delta_max <- Rmlx::mlx_maximum(delta_max, max(abs(delta_mlx)))

      t_next <- (one_mlx + sqrt(one_mlx + four_mlx * t_prev_local^2)) / 2
      z_mlx <- beta_new_mlx + ((t_prev_local - one_mlx) / t_next) * delta_mlx
      beta_mlx <- beta_new_mlx
      t_prev_local <- t_next
    }

    list(
      beta = beta_mlx,
      z = z_mlx,
      t_prev = t_prev_local,
      delta_max = delta_max
    )
  })
}

#' Compile binomial glmnet updates
#'
#' Factory for logistic elastic-net path updates. The returned function captures
#' both the chunk length and whether the intercept is updated.
#'
#' @param n_steps Number of proximal-gradient steps in the compiled chunk.
#' @param fit_intercept Logical; include intercept updates in the traced body.
#' @return Compiled function that updates binomial path state.
#' @noRd
.mlxs_glmnet_make_binomial_compiled_chunk <- function(n_steps,
                                                      fit_intercept) {
  Rmlx::mlx_compile(function(x_mlx,
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
                             zero_mlx) {
    delta_max <- zero_mlx
    intercept_delta_max <- zero_mlx

    for (i in seq_len(n_steps)) {
      grad_mlx <- crossprod(x_mlx, residual_mlx) / n_obs_mlx
      grad_mlx <- grad_mlx + beta_mlx * ridge_penalty_mlx

      beta_new_mlx <- .mlxs_soft_threshold(
        beta_mlx - step_mlx * grad_mlx,
        thresh_mlx
      )
      delta_mlx <- beta_new_mlx - beta_mlx

      if (fit_intercept) {
        intercept_delta_mlx <- step_mlx * (Rmlx::mlx_sum(residual_mlx) /
          n_obs_mlx)
        intercept_mlx <- intercept_mlx - intercept_delta_mlx
        eta_mlx <- eta_mlx + x_mlx %*% delta_mlx -
          ones_mlx * intercept_delta_mlx
        intercept_delta_max <- Rmlx::mlx_maximum(
          intercept_delta_max,
          abs(intercept_delta_mlx)
        )
      } else {
        eta_mlx <- eta_mlx + x_mlx %*% delta_mlx
      }

      mu_mlx <- 1 / (1 + exp(-eta_mlx))
      residual_mlx <- mu_mlx - y_mlx
      beta_mlx <- beta_new_mlx

      delta_max <- Rmlx::mlx_maximum(delta_max, max(abs(delta_mlx)))
    }

    list(
      beta = beta_mlx,
      intercept = intercept_mlx,
      eta = eta_mlx,
      residual = residual_mlx,
      delta_max = delta_max,
      intercept_delta_max = intercept_delta_max
    )
  })
}
