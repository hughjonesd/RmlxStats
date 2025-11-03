.mlxs_as_mlx <- function(x) {
  if (inherits(x, "mlx")) {
    x
  } else {
    Rmlx::as_mlx(x)
  }
}

.mlxs_to_numeric <- function(x) {
  if (inherits(x, "mlx")) {
    as.numeric(as.matrix(x))
  } else {
    as.numeric(x)
  }
}
