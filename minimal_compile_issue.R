#!/usr/bin/env Rscript
# Minimal reproducible example: mlx_compile() loses list structure

library(Rmlx)

# Simple function returning a list
test_func <- function(x, y) {
  sum_result <- x + y
  product_result <- x * y
  list(sum = sum_result, product = product_result)
}

# Create test data
x <- as_mlx(matrix(1:5, ncol = 1))
y <- as_mlx(matrix(6:10, ncol = 1))

cat("=== Uncompiled version ===\n")
result_uncompiled <- test_func(x, y)
cat("Class:", class(result_uncompiled), "\n")
cat("Names:", names(result_uncompiled), "\n")
cat("sum class:", class(result_uncompiled$sum), "\n")
cat("product class:", class(result_uncompiled$product), "\n")
cat("sum values:", as.numeric(result_uncompiled$sum), "\n\n")

cat("=== Compiled version ===\n")
test_func_compiled <- mlx_compile(test_func)
result_compiled <- test_func_compiled(x, y)
cat("Class:", class(result_compiled), "\n")
cat("Names:", names(result_compiled), "\n")
cat("Length:", length(result_compiled), "\n")

if (length(names(result_compiled)) == 0) {
  cat("\n*** BUG: Compiled version returns list with no names ***\n")
  cat("Cannot access result$sum or result$product\n")
} else {
  cat("sum class:", class(result_compiled$sum), "\n")
  cat("product class:", class(result_compiled$product), "\n")
}
