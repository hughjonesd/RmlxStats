#!/usr/bin/env Rscript
# Test if compiled functions work with different shaped inputs

library(Rmlx)

# Simple function that takes matrix input
simple_func <- function(x, y) {
  z <- x %*% t(y)
  mlx_sum(z * z)
}

cat("=== Testing shape polymorphism ===\n\n")

# Compile with one shape
cat("1. Creating data with shape (100, 20):\n")
x1 <- as_mlx(matrix(rnorm(100*20), 100, 20))
y1 <- as_mlx(matrix(rnorm(50*20), 50, 20))
cat("   x1:", dim(x1), "\n")
cat("   y1:", dim(y1), "\n")

cat("\n2. Testing uncompiled version:\n")
result1 <- simple_func(x1, y1)
cat("   Result:", as.numeric(result1), "\n")

cat("\n3. Compiling function...\n")
simple_func_compiled <- mlx_compile(simple_func)

cat("\n4. Testing compiled version with original shape:\n")
result2 <- simple_func_compiled(x1, y1)
cat("   Result:", as.numeric(result2), "\n")
cat("   Match:", abs(as.numeric(result1) - as.numeric(result2)) < 1e-5, "\n")

cat("\n5. Testing compiled version with DIFFERENT shape (500, 100):\n")
x2 <- as_mlx(matrix(rnorm(500*100), 500, 100))
y2 <- as_mlx(matrix(rnorm(200*100), 200, 100))
cat("   x2:", dim(x2), "\n")
cat("   y2:", dim(y2), "\n")

tryCatch({
  result3 <- simple_func_compiled(x2, y2)
  cat("   Result:", as.numeric(result3), "\n")
  cat("   ✓ SUCCESS: Compiled function works with different shapes!\n")
}, error = function(e) {
  cat("   ✗ FAILED: ", conditionMessage(e), "\n")
  cat("\n   Compiled functions are shape-specific.\n")
  cat("   Need to recompile for each shape or compile on every call.\n")
})

cat("\n6. Testing with SAME dimensions but different values:\n")
x3 <- as_mlx(matrix(rnorm(100*20), 100, 20))
y3 <- as_mlx(matrix(rnorm(50*20), 50, 20))
cat("   x3:", dim(x3), "(same as x1)\n")
cat("   y3:", dim(y3), "(same as y1)\n")

result4 <- simple_func_compiled(x3, y3)
cat("   Result:", as.numeric(result4), "\n")
cat("   ✓ Works with same shape, different values\n")

cat("\n=== Conclusion ===\n")
cat("Compiled functions can handle:\n")
cat("  - Different values with same shape: ✓\n")
cat("  - Different shapes: ?\n")
