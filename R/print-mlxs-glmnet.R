#' @export
print.mlxs_glmnet <- function(
  x,
  n_lambda = 5L,
  digits = getOption("digits"),
  ...
) {
  cat(
    "MLX elastic net fit (family = ",
    x$family,
    ", alpha = ",
    x$alpha,
    ")\n",
    sep = ""
  )

  lambda_vals <- x$lambda_numeric
  if (length(lambda_vals)) {
    k <- min(n_lambda, length(lambda_vals))
    idx <- seq_len(k)
    lambda_fmt <- format(signif(lambda_vals[idx], digits = digits))

    beta_slice <- x$beta[, idx, drop = FALSE]
    beta_host <- as.matrix(beta_slice)
    nnz <- colSums(abs(beta_host) > 0)

    cat("  Lambdas: ", paste(lambda_fmt, collapse = ", "), "\n", sep = "")
    cat("  Nonzero coefficients: ", paste(nnz, collapse = ", "), "\n", sep = "")
  } else {
    cat("  (No lambdas stored)\n")
  }

  invisible(x)
}
