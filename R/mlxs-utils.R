# Internal helpers for MLX-aware objects

.mlxs_as_numeric <- function(x) {
  if (is.null(x)) {
    return(x)
  }
  if (inherits(x, "mlx")) {
    vals <- as.vector(as.matrix(x))
    if (!is.numeric(vals)) {
      vals <- as.numeric(vals)
    }
    return(vals)
  }
  x
}

.mlxs_as_matrix <- function(x) {
  if (is.null(x)) {
    return(x)
  }
  if (inherits(x, "mlx")) {
    as.matrix(x)
  } else {
    x
  }
}
