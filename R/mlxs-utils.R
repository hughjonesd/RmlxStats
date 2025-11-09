# Internal helpers for MLX-aware objects

.mlxs_as_numeric <- function(x) {
  if (is.null(x)) {
    return(x)
  }
  if (inherits(x, "mlx")) {
    drop(as.matrix(x))
  } else {
    x
  }
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
