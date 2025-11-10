.mlxs_bootstrap_collect <- function(B, n, seed, progress, build_boot) {
  if (!is.null(seed)) {
    old_seed <- .Random.seed
    on.exit(assign(".Random.seed", old_seed, envir = .GlobalEnv), add = TRUE)
    set.seed(seed)
  }
  coef_stack <- vector("list", B)
  pb <- NULL
  if (isTRUE(progress)) {
    pb <- utils::txtProgressBar(min = 0, max = B, style = 3)
    on.exit(close(pb), add = TRUE)
  }
  for (rep_idx in seq_len(B)) {
    idx <- sample.int(n, n, replace = TRUE)
    coef_stack[[rep_idx]] <- build_boot(idx)
    if (!is.null(pb)) {
      utils::setTxtProgressBar(pb, rep_idx)
    }
  }
  coef_stack
}

.mlxs_bootstrap_finalize <- function(coef_stack, coef_names, method, B, seed) {
  coef_array <- Rmlx::mlx_stack(coef_stack, axis = 3L)
  se_mlx <- Rmlx::mlx_std(coef_array, axis = 3L, drop = FALSE, ddof = 1L)
  se_mlx <- Rmlx::mlx_reshape(se_mlx, c(length(coef_names), 1L))
  list(se = se_mlx, samples = NULL, B = B, seed = seed, method = method)
}
