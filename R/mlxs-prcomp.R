#' MLX-backed principal components analysis
#'
#' Perform principal components analysis with MLX arrays, keeping the centred
#' and scaled data on device throughout the decomposition.
#'
#' The interface follows [stats::prcomp()] closely. Full-rank fits use an exact
#' decomposition. When `rank.` is supplied and smaller than `min(nrow(x),
#' ncol(x))`, a randomized truncated PCA path is used instead.
#'
#' @param x Numeric matrix-like object or MLX array with observations in rows.
#' @param retx Should the rotated scores be returned?
#' @param center,scale. Passed to [base::scale()]. User-supplied vectors are
#'   supported.
#' @param tol Optional tolerance for omitting components with small standard
#'   deviations, relative to the leading component.
#' @param rank. Optional maximal rank. If smaller than `min(n, p)`, the fit
#'   uses the randomized truncated PCA path.
#' @param oversample Oversampling added to the randomized subspace dimension.
#'   Ignored for exact fits.
#' @param n_iter Number of randomized power iterations. Ignored for exact fits.
#' @param seed Seed used for the randomized projection basis. Ignored for exact
#'   fits.
#' @param ... Additional arguments are rejected for compatibility with
#'   [stats::prcomp()].
#' @return An object of class `c("mlxs_prcomp", "prcomp")`.
#' @export
mlxs_prcomp <- function(x,
                        retx = TRUE,
                        center = TRUE,
                        scale. = FALSE,
                        tol = NULL,
                        rank. = NULL,
                        oversample = 10L,
                        n_iter = 2L,
                        seed = 1L,
                        ...) {
  if (length(list(...)) > 0L) {
    stop("Unused arguments in mlxs_prcomp().", call. = FALSE)
  }

  x_names <- if (inherits(x, "mlx")) NULL else dimnames(as.matrix(x))
  x_mlx <- if (inherits(x, "mlx")) Rmlx::as_mlx(x) else Rmlx::as_mlx(as.matrix(x))

  if (length(dim(x_mlx)) != 2L) {
    stop("x must be a 2D matrix-like object.", call. = FALSE)
  }
  if (any(!is.finite(x_mlx))) {
    stop("x must contain only finite values.", call. = FALSE)
  }

  n_obs <- nrow(x_mlx)
  n_pred <- ncol(x_mlx)
  rank_limit <- .mlxs_prcomp_rank_limit(rank., n_obs, n_pred)
  tol <- .mlxs_prcomp_validate_tol(tol)
  oversample <- .mlxs_prcomp_validate_count(oversample, "oversample")
  n_iter <- .mlxs_prcomp_validate_count(n_iter, "n_iter")
  seed <- .mlxs_prcomp_validate_seed(seed)

  x_scaled <- scale(x_mlx, center = center, scale = scale.)
  x_center <- .mlxs_prcomp_param_to_mlx(
    attr(x_scaled, "scaled:center"),
    n_pred,
    x_scaled
  )
  x_scale <- .mlxs_prcomp_param_to_mlx(
    attr(x_scaled, "scaled:scale"),
    n_pred,
    x_scaled
  )

  if (!identical(x_scale, FALSE) && any(x_scale == 0)) {
    stop("cannot rescale a constant/zero column to unit variance",
         call. = FALSE)
  }

  full_rank <- min(n_obs, n_pred)
  fit <- if (is.null(rank.) || rank_limit == full_rank) {
    .mlxs_prcomp_exact(
      x_scaled = x_scaled,
      rank_limit = rank_limit,
      tol = tol,
      retx = retx
    )
  } else {
    .mlxs_prcomp_randomized(
      x_scaled = x_scaled,
      rank_limit = rank_limit,
      tol = tol,
      retx = retx,
      oversample = oversample,
      n_iter = n_iter,
      seed = seed
    )
  }

  component_names <- paste0("PC", seq_len(fit$rank))
  result <- list(
    sdev = fit$sdev,
    rotation = fit$rotation,
    center = x_center,
    scale = x_scale,
    rank = fit$rank,
    method = fit$method,
    call = match.call(),
    feature_names = x_names[[2L]],
    observation_names = x_names[[1L]],
    component_names = component_names,
    n_obs = n_obs,
    n_features = n_pred
  )

  if (retx) {
    result$x <- fit$x
  }

  class(result) <- c("mlxs_prcomp", "prcomp")
  result
}

.mlxs_prcomp_exact <- function(x_scaled,
                               rank_limit,
                               tol,
                               retx) {
  n_obs <- nrow(x_scaled)
  n_pred <- ncol(x_scaled)

  if (n_obs < n_pred) {
    return(.mlxs_prcomp_exact_wide(
      x_scaled = x_scaled,
      rank_limit = rank_limit,
      tol = tol,
      retx = retx
    ))
  }

  denom <- max(1L, n_obs - 1L)
  cov_mat <- crossprod(x_scaled) / denom
  eig <- Rmlx::mlx_eigh(cov_mat)

  idx <- seq.int(from = n_pred, to = 1L)
  values <- eig$values[idx]
  rotation <- eig$vectors[, idx, drop = FALSE]
  values <- Rmlx::mlx_maximum(
    values,
    Rmlx::as_mlx(0, dtype = Rmlx::mlx_dtype(values), device = values$device)
  )
  sdev_all <- sqrt(values)
  keep <- .mlxs_prcomp_keep_count(sdev_all, rank_limit, tol)

  rotation <- rotation[, seq_len(keep), drop = FALSE]
  sdev <- sdev_all[seq_len(keep)]
  scores <- if (retx) x_scaled %*% rotation else NULL

  .mlxs_prcomp_finalize(
    sdev = sdev,
    rotation = rotation,
    scores = scores,
    n_obs = n_obs,
    n_features = n_pred,
    method = "exact"
  )
}

.mlxs_prcomp_exact_wide <- function(x_scaled,
                                    rank_limit,
                                    tol,
                                    retx) {
  n_obs <- nrow(x_scaled)
  n_pred <- ncol(x_scaled)
  denom <- max(1L, n_obs - 1L)
  gram <- tcrossprod(x_scaled) / denom
  eig <- Rmlx::mlx_eigh(gram)

  idx <- seq.int(from = n_obs, to = 1L)
  values <- eig$values[idx]
  left_vecs <- eig$vectors[, idx, drop = FALSE]
  values <- Rmlx::mlx_maximum(
    values,
    Rmlx::as_mlx(0, dtype = Rmlx::mlx_dtype(values), device = values$device)
  )
  sdev_all <- sqrt(values)
  keep <- .mlxs_prcomp_keep_count(sdev_all, rank_limit, tol)

  if (keep == 0L) {
    return(.mlxs_prcomp_finalize(
      sdev = Rmlx::as_mlx(numeric()),
      rotation = Rmlx::as_mlx(matrix(numeric(), n_pred, 0L)),
      scores = if (retx) Rmlx::as_mlx(matrix(numeric(), n_obs, 0L)) else NULL,
      n_obs = n_obs,
      n_features = n_pred,
      method = "exact"
    ))
  }

  zero_tol <- sdev_all[1] * 1e-6
  n_positive <- as.integer(sum(sdev_all[seq_len(keep)] > zero_tol))
  if (n_positive < keep) {
    return(.mlxs_prcomp_exact_svd(
      x_scaled = x_scaled,
      rank_limit = rank_limit,
      tol = tol,
      retx = retx
    ))
  }

  sdev <- sdev_all[seq_len(keep)]
  left_vecs <- left_vecs[, seq_len(keep), drop = FALSE]
  d_vals <- sdev * sqrt(denom)
  rotation <- (t(x_scaled) %*% left_vecs) /
    Rmlx::mlx_reshape(d_vals, c(1L, keep))
  scores <- if (retx) {
    left_vecs * Rmlx::mlx_reshape(d_vals, c(1L, keep))
  } else {
    NULL
  }

  .mlxs_prcomp_finalize(
    sdev = sdev,
    rotation = rotation,
    scores = scores,
    n_obs = n_obs,
    n_features = n_pred,
    method = "exact"
  )
}

.mlxs_prcomp_exact_svd <- function(x_scaled,
                                   rank_limit,
                                   tol,
                                   retx) {
  n_obs <- nrow(x_scaled)
  n_pred <- ncol(x_scaled)
  full_rank <- min(n_obs, n_pred)
  decomp <- Rmlx::svd(x_scaled, nu = 0L, nv = full_rank)
  sdev_all <- decomp$d / sqrt(max(1L, n_obs - 1L))
  keep <- .mlxs_prcomp_keep_count(sdev_all, rank_limit, tol)

  rotation <- decomp$v[, seq_len(keep), drop = FALSE]
  sdev <- sdev_all[seq_len(keep)]
  scores <- if (retx) x_scaled %*% rotation else NULL

  .mlxs_prcomp_finalize(
    sdev = sdev,
    rotation = rotation,
    scores = scores,
    n_obs = n_obs,
    n_features = n_pred,
    method = "exact"
  )
}

.mlxs_prcomp_randomized <- function(x_scaled,
                                    rank_limit,
                                    tol,
                                    retx,
                                    oversample,
                                    n_iter,
                                    seed) {
  n_obs <- nrow(x_scaled)
  n_pred <- ncol(x_scaled)
  work_rank <- min(n_pred, rank_limit + oversample)
  omega <- .mlxs_prcomp_random_normal(
    dim = c(n_pred, work_rank),
    seed = seed,
    dtype = Rmlx::mlx_dtype(x_scaled),
    device = x_scaled$device
  )
  q_basis <- qr(omega)$Q[, seq_len(work_rank), drop = FALSE]

  if (n_iter > 0L) {
    for (iter in seq_len(n_iter)) {
      q_basis <- .mlxs_prcomp_power_step_runner(x_scaled, q_basis)
    }
  }

  scores_basis <- x_scaled %*% q_basis
  small_cov <- crossprod(scores_basis) / max(1L, n_obs - 1L)
  eig <- Rmlx::mlx_eigh(small_cov)
  idx <- seq.int(from = work_rank, to = 1L)
  values <- eig$values[idx]
  basis_rot <- eig$vectors[, idx, drop = FALSE]
  values <- Rmlx::mlx_maximum(
    values,
    Rmlx::as_mlx(0, dtype = Rmlx::mlx_dtype(values), device = values$device)
  )
  sdev_all <- sqrt(values)
  keep <- .mlxs_prcomp_keep_count(sdev_all, rank_limit, tol)

  basis_rot <- basis_rot[, seq_len(keep), drop = FALSE]
  sdev <- sdev_all[seq_len(keep)]
  rotation <- q_basis %*% basis_rot
  scores <- if (retx) scores_basis %*% basis_rot else NULL

  .mlxs_prcomp_finalize(
    sdev = sdev,
    rotation = rotation,
    scores = scores,
    n_obs = n_obs,
    n_features = n_pred,
    method = "randomized"
  )
}

.mlxs_prcomp_power_step_runner <- local({
  compiled <- NULL

  function(x_scaled, q_basis) {
    if (is.null(compiled)) {
      compiled <<- Rmlx::mlx_compile(
        function(x_scaled, q_basis) {
          qr(t(x_scaled) %*% (x_scaled %*% q_basis))$Q
        }
      )
    }

    compiled(x_scaled, q_basis)
  }
})

.mlxs_prcomp_finalize <- function(sdev,
                                  rotation,
                                  scores,
                                  n_obs,
                                  n_features,
                                  method) {
  rank <- length(sdev)

  if (rank == 0L) {
    return(list(
      sdev = sdev,
      rotation = Rmlx::as_mlx(matrix(numeric(), n_features, 0L)),
      x = if (is.null(scores)) NULL else Rmlx::as_mlx(matrix(numeric(), n_obs, 0L)),
      rank = 0L,
      method = method
    ))
  }

  signs <- .mlxs_prcomp_column_signs(rotation)
  rotation <- rotation * signs
  if (!is.null(scores)) {
    scores <- scores * signs
  }

  list(
    sdev = sdev,
    rotation = rotation,
    x = scores,
    rank = rank,
    method = method
  )
}

.mlxs_prcomp_column_signs <- function(rotation) {
  max_idx <- Rmlx::mlx_argmax(abs(rotation), axis = 1L, drop = FALSE)
  signs <- Rmlx::mlx_take_along_axis(sign(rotation), max_idx, axis = 1L)
  signs + (signs == 0)
}

.mlxs_prcomp_keep_count <- function(sdev_all, rank_limit, tol) {
  keep <- rank_limit
  if (!is.null(tol) && keep > 0L) {
    cutoff <- sdev_all[1] * tol
    keep <- min(keep, as.integer(sum(sdev_all[seq_len(keep)] > cutoff)))
  }
  keep
}

.mlxs_prcomp_rank_limit <- function(rank., n_obs, n_pred) {
  full_rank <- min(n_obs, n_pred)
  if (is.null(rank.)) {
    return(full_rank)
  }

  if (length(rank.) != 1L ||
      !is.finite(rank.) ||
      as.integer(rank.) <= 0L) {
    stop("rank. must be a single positive finite integer.", call. = FALSE)
  }

  min(as.integer(rank.), full_rank)
}

.mlxs_prcomp_validate_tol <- function(tol) {
  if (is.null(tol)) {
    return(NULL)
  }
  if (length(tol) != 1L || !is.finite(tol) || tol < 0) {
    stop("tol must be NULL or a single non-negative finite number.",
         call. = FALSE)
  }
  tol
}

.mlxs_prcomp_validate_count <- function(value, name) {
  if (length(value) != 1L || !is.finite(value) || as.integer(value) < 0L) {
    stop(name, " must be a single non-negative integer.", call. = FALSE)
  }
  as.integer(value)
}

.mlxs_prcomp_validate_seed <- function(seed) {
  if (length(seed) != 1L || !is.finite(seed)) {
    stop("seed must be a single finite number.", call. = FALSE)
  }
  as.numeric(seed)
}

.mlxs_prcomp_param_to_mlx <- function(value, n_pred, x_scaled) {
  if (is.null(value)) {
    return(FALSE)
  }
  if (inherits(value, "mlx")) {
    return(value)
  }

  value <- as.numeric(value)
  Rmlx::as_mlx(
    matrix(value, nrow = 1L, ncol = n_pred),
    dtype = Rmlx::mlx_dtype(x_scaled),
    device = x_scaled$device
  )
}

.mlxs_prcomp_random_normal <- function(dim, seed, dtype, device) {
  base_key <- Rmlx::mlx_key(seed)
  subkeys <- Rmlx::mlx_key_split(base_key, num = 2L)
  u1_bits <- Rmlx::mlx_key_bits(dim, key = subkeys[[1L]], device = device)
  u2_bits <- Rmlx::mlx_key_bits(dim, key = subkeys[[2L]], device = device)

  scale <- Rmlx::as_mlx(4294967296, dtype = dtype, device = device)
  eps <- Rmlx::as_mlx(1e-7, dtype = dtype, device = device)
  two_pi <- Rmlx::as_mlx(2 * pi, dtype = dtype, device = device)

  u1 <- (Rmlx::mlx_cast(u1_bits, dtype = dtype) + 0.5) / scale
  u2 <- (Rmlx::mlx_cast(u2_bits, dtype = dtype) + 0.5) / scale
  u1 <- Rmlx::mlx_maximum(u1, eps)

  sqrt(-2 * log(u1)) * cos(two_pi * u2)
}

#' PCA methods for `mlxs_prcomp`
#'
#' `predict.mlxs_prcomp()` returns MLX scores. The presentation methods
#' (`print()`, `summary()`, `plot()`, and `biplot()`) reuse the base `prcomp`
#' implementations by converting to a temporary host-backed `prcomp` object.
#'
#' @param object,x A fitted `mlxs_prcomp` object.
#' @param newdata Optional new observations to project.
#' @param data Optional original data to append PCA scores to in
#'   `augment.mlxs_prcomp()`.
#' @param output Output format for `augment.mlxs_prcomp()`: either a data frame
#'   with appended score columns or the MLX score matrix directly.
#' @param ... Passed through to the corresponding base method.
#' @return Method-specific output. `predict.mlxs_prcomp()` returns an MLX
#'   matrix. `augment.mlxs_prcomp()` returns either a data frame or MLX matrix.
#' @name mlxs-prcomp-methods
NULL

.mlxs_prcomp_as_prcomp <- function(x) {
  rotation <- as.matrix(x$rotation)
  dimnames(rotation) <- list(x$feature_names, x$component_names)

  result <- list(
    sdev = stats::setNames(as.numeric(x$sdev), x$component_names),
    rotation = rotation,
    center = .mlxs_prcomp_host_vector(x$center, x$feature_names),
    scale = .mlxs_prcomp_host_vector(x$scale, x$feature_names)
  )

  if (!is.null(x$x)) {
    scores <- as.matrix(x$x)
    dimnames(scores) <- list(x$observation_names, x$component_names)
    result$x <- scores
  }

  class(result) <- "prcomp"
  result
}

.mlxs_prcomp_host_vector <- function(x, names) {
  if (identical(x, FALSE)) {
    return(FALSE)
  }

  values <- as.numeric(x)
  if (!is.null(names) && length(values) == length(names)) {
    stats::setNames(values, names)
  } else {
    values
  }
}

#' @export
#' @rdname mlxs-prcomp-methods
predict.mlxs_prcomp <- function(object, newdata, ...) {
  if (missing(newdata)) {
    if (!is.null(object$x)) {
      return(object$x)
    }
    stop("no scores are available: refit with 'retx=TRUE'", call. = FALSE)
  }

  newdata_names <- NULL
  x_mlx <- if (inherits(newdata, "mlx")) {
    Rmlx::as_mlx(newdata)
  } else {
    newdata_mat <- as.matrix(newdata)
    newdata_names <- colnames(newdata_mat)
    Rmlx::as_mlx(newdata_mat)
  }

  if (length(dim(x_mlx)) != 2L) {
    stop("newdata must be a 2D matrix-like object.", call. = FALSE)
  }

  feature_names <- object$feature_names
  if (!is.null(feature_names) && !is.null(newdata_names)) {
    if (!all(feature_names %in% newdata_names)) {
      stop(
        "'newdata' does not have named columns matching one or more of the original columns",
        call. = FALSE
      )
    }
    idx <- match(feature_names, newdata_names)
    x_mlx <- x_mlx[, idx, drop = FALSE]
  } else if (ncol(x_mlx) != nrow(object$rotation)) {
    stop("'newdata' does not have the correct number of columns",
         call. = FALSE)
  }

  if (!identical(object$center, FALSE)) {
    x_mlx <- x_mlx - object$center
  }
  if (!identical(object$scale, FALSE)) {
    x_mlx <- x_mlx / object$scale
  }

  x_mlx %*% object$rotation
}

#' @export
#' @rdname mlxs-prcomp-methods
print.mlxs_prcomp <- function(x, ...) {
  print(.mlxs_prcomp_as_prcomp(x), ...)
  invisible(x)
}

#' @export
#' @rdname mlxs-prcomp-methods
summary.mlxs_prcomp <- function(object, ...) {
  summary(.mlxs_prcomp_as_prcomp(object), ...)
}

#' @export
#' @rdname mlxs-prcomp-methods
plot.mlxs_prcomp <- function(x, ...) {
  graphics::plot(.mlxs_prcomp_as_prcomp(x), ...)
}

#' @export
#' @rdname mlxs-prcomp-methods
biplot.mlxs_prcomp <- function(x, ...) {
  stats::biplot(.mlxs_prcomp_as_prcomp(x), ...)
}

#' @export
#' @rdname mlxs-prcomp-methods
nobs.mlxs_prcomp <- function(object, ...) {
  object$n_obs
}

#' @export
#' @rdname mlxs-prcomp-methods
tidy.mlxs_prcomp <- function(x, ...) {
  sum_obj <- summary(x, ...)
  importance <- t(sum_obj$importance)
  data.frame(
    component = rownames(importance),
    std.dev = importance[, "Standard deviation"],
    proportion = importance[, "Proportion of Variance"],
    cumulative = importance[, "Cumulative Proportion"],
    row.names = NULL,
    check.names = FALSE
  )
}

#' @export
#' @rdname mlxs-prcomp-methods
augment.mlxs_prcomp <- function(x,
                                data = NULL,
                                newdata = NULL,
                                output = c("data.frame", "mlx"),
                                ...) {
  output <- match.arg(output)
  scores <- if (is.null(newdata)) predict(x) else predict(x, newdata = newdata)

  if (output == "mlx") {
    return(scores)
  }

  scores_df <- as.data.frame(as.matrix(scores))
  names(scores_df) <- paste0(".fitted", x$component_names)

  base_data <- if (is.null(newdata)) data else newdata
  if (is.null(base_data)) {
    out <- as.data.frame(matrix(nrow = nrow(scores_df), ncol = 0L))
  } else {
    out <- as.data.frame(base_data)
  }

  cbind(out, scores_df, row.names = NULL)
}
