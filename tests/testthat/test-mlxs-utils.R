test_that(".mlxs_as_numeric flattens mlx arrays to numeric vectors", {
  bool_mlx <- Rmlx::mlx_matrix(rep(TRUE, 4), nrow = 2, ncol = 2)
  num_vec <- .mlxs_as_numeric(bool_mlx)
  expect_type(num_vec, "double")
  expect_equal(num_vec, rep(1, 4))
})

test_that("mlx_matrix insists on at least one explicit dimension", {
  expect_error(
    Rmlx::mlx_matrix(1:4),
    "Provide either nrow or ncol when calling mlx_matrix",
    fixed = FALSE
  )
})
