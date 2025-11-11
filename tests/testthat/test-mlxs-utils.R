test_that("mlx_matrix insists on at least one explicit dimension", {
  expect_error(
    Rmlx::mlx_matrix(1:4),
    "Provide either nrow or ncol when calling mlx_matrix",
    fixed = FALSE
  )
})
