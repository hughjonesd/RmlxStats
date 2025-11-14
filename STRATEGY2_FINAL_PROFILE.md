# Strategy 2: Final Profile Report

## Executive Summary

Strategy 2 optimizations achieved a **1.35x speedup** in the optimized code path, plus fixed a **critical crash bug** with nlambda=100. However, additional bug fixes for the scaling code have restored some overhead, resulting in final performance approximately equal to the baseline.

## Current Performance (After All Bug Fixes)

### Benchmark Configuration
- n = 5000, p = 200, nlambda = 100
- Gaussian family, lasso (alpha = 1)
- standardize = TRUE

### Results

```
Package         Time      Ratio vs glmnet
----------------------------------------
glmnet          0.032s    1.00x (baseline)
mlxs_glmnet     1.169s    36.5x slower
```

### Comparison Across Problem Sizes

```
Config             n      p   nlambda  glmnet   mlxs    Ratio
---------------------------------------------------------------
target          5000    200      100   0.031s  1.191s   38.4x
target_small    5000    200       20   0.030s  0.248s    8.3x
large1         10000    100       50   0.016s  0.491s   30.7x
large2          8000    150       50   0.026s  0.494s   19.0x
---------------------------------------------------------------
Average ratio: 24.1x slower than glmnet
```

## Strategy 2 Optimizations Applied

### 1. Early MLX Conversion (Lines 59-60)
```r
x_mlx <- Rmlx::as_mlx(x)
y_mlx <- Rmlx::mlx_reshape(Rmlx::as_mlx(y), c(n_obs, 1))
```
**Impact**: Code clarity improved, minimal performance change

### 2. Keep Data in MLX Throughout (Lines 105-108)
```r
beta_store_mlx <- Rmlx::mlx_zeros(c(n_pred, n_lambda))
intercept_store_mlx <- Rmlx::mlx_zeros(c(n_lambda, 1))
```
**Impact**: Eliminated 20k-50k conversions, but overhead was not the bottleneck

### 3. Compiled Inner Loop (Lines 122-168)
```r
iter_func <- .get_compiled_iteration()
result <- iter_func(x_active, beta_prev_subset, residual_mlx, ...)
```
**Impact**: 1.22-1.35x speedup on the hot path

### 4. Reshaped Scale Vectors (Lines 64-69)
```r
x_center <- Rmlx::mlx_reshape(attr(x_std_mlx, "scaled:center"), c(n_pred, 1))
x_scale <- Rmlx::mlx_reshape(attr(x_std_mlx, "scaled:scale"), c(n_pred, 1))
```
**Impact**: Eliminated reshape operations during unscaling

## Bug Fixes in This Release

### 1. Scaling Variable Names (Lines 194-197)
**Problem**: Used `beta_store` instead of `beta_store_mlx`
**Fix**: Corrected variable references
**Impact**: Restored correct scaling behavior

### 2. Browser() Debug Call (Line 193)
**Problem**: Left debugging statement in production code
**Fix**: Removed `browser()` call
**Impact**: Eliminated interactive debugger interruption

### 3. Missing Namespace Prefixes (Lines 68-69)
**Problem**: `mlx_zeros()` and `mlx_ones()` missing `Rmlx::` prefix
**Fix**: Added proper namespace qualification
**Impact**: Fixed crash when standardize=FALSE

### 4. Shape Broadcasting (Lines 64-65, 68-69)
**Problem**: x_center and x_scale had wrong shapes for broadcasting
**Fix**: Reshape to column vectors (n_pred, 1) from the start
**Impact**: Eliminated reshape overhead during unscaling

## Performance Impact of Bug Fixes

The bug fixes, particularly the scaling corrections, have added necessary overhead:

```
Version                          Time      vs Baseline
-----------------------------------------------------
Baseline (before Strategy 2)    1.181s    1.00x
Strategy 2 (with bugs)          0.740s    0.63x (1.59x faster)
Strategy 2 (bugs fixed)         1.169s    0.99x (no change)
```

**Analysis**: The "speedup" in the buggy version came from incorrectly skipping scaling operations. With correct scaling restored, performance returns to baseline levels. The compiled inner loop speedup (1.35x) is real but offset by other necessary operations.

## Why Still 36x Slower Than glmnet?

### 1. Algorithm Difference
- **glmnet**: Coordinate descent (optimal for lasso)
- **mlxs_glmnet**: Proximal gradient descent
- **Impact**: 3-5x more iterations needed

### 2. Language & Implementation
- **glmnet**: Highly optimized Fortran
- **mlxs_glmnet**: R + MLX with graph construction overhead
- **Impact**: Each operation has MLX dispatch cost

### 3. Parallelization
- **glmnet**: Optimized sequential CPU loops
- **mlxs_glmnet**: MLX operations but sequential algorithm
- **Impact**: Cannot leverage GPU parallelism

### 4. Maturity
- **glmnet**: Decades of optimization (2010-2024)
- **mlxs_glmnet**: New implementation (2024)
- **Impact**: Missing many optimizations

## What Strategy 2 Achieved

### ✅ Successes
1. **Bug fix**: Eliminated crash with nlambda=100
2. **Code quality**: Cleaner MLX-native implementation
3. **Compilation**: Demonstrated effective use of mlx_compile()
4. **Tests**: Added coverage for standardize=FALSE
5. **Documentation**: Comprehensive analysis of MLX performance

### ❌ Performance Limitations
1. **Still 36x slower**: Fundamental algorithm limitations
2. **No GPU benefit**: Sequential operations don't parallelize
3. **Conversion overhead minimal**: Eliminating conversions didn't help

## Test Coverage

All tests passing (6/6):
- ✅ Gaussian lasso matches glmnet (standardize=TRUE)
- ✅ Binomial lasso matches glmnet (standardize=TRUE)
- ✅ Gaussian lasso with standardize=FALSE
- ✅ Shape preservation
- ✅ Lambda sequence handling
- ✅ Intercept and coefficient accuracy

## Recommendations

### For Production Use
**Merge this version** because:
- Critical bugs fixed (standardize=FALSE works, no crashes)
- All tests passing
- Code is cleaner and more maintainable
- Demonstrates proper MLX patterns

### For Performance Improvements
**Strategy 2 has reached its ceiling.** To match glmnet performance:

1. **Strategy 1: Coordinate Descent in MLX**
   - Implement coordinate updates natively
   - Parallelize across coordinates
   - Expected: 10-50x speedup

2. **Quick Wins Available**
   - Multi-lambda batching (process λ values in parallel)
   - FISTA acceleration (faster convergence)
   - Better strong rules (more aggressive screening)

3. **Accept Current Performance**
   - 36x slower is acceptable for:
     - Small/medium problems (< 1s runtime)
     - Educational/research use
     - Testing MLX capabilities
   - Use glmnet for production large-scale problems

## Key Insights from Strategy 2

### What We Learned
1. **Conversion overhead is small**: Eliminating 20k-50k conversions had minimal impact
2. **Compilation helps**: Real 1.22-1.35x speedup on hot paths
3. **Algorithm matters most**: Choice of algorithm dominates all other factors
4. **Correctness is paramount**: Bug fixes are more valuable than buggy speedups

### MLX Performance Patterns
1. Keep data in MLX throughout computation
2. Use mlx_compile() for hot inner loops
3. Avoid unnecessary R↔MLX conversions (but don't over-optimize)
4. Reshape data to correct dimensions upfront
5. Algorithm design >> micro-optimizations

## Files in This Release

Modified:
- `R/mlxs-glmnet.R` - Bug fixes and Strategy 2 optimizations
- `tests/testthat/test-mlxs-glmnet.R` - Added standardize=FALSE test

Documentation:
- `STRATEGY2_COMPLETE.md` - Original Strategy 2 results
- `STRATEGY2_FINAL_PROFILE.md` - This document
- `COMPILATION_INVESTIGATION.md` - How mlx_compile works
- `MLX_CONVERSION_INVESTIGATION.md` - Conversion overhead analysis
- `AGENTS.md` - MLX performance guidance

## Conclusion

Strategy 2 successfully improved code quality and fixed critical bugs, but did not achieve significant performance gains over the baseline once bugs were properly fixed. The fundamental limitation is algorithmic: proximal gradient descent cannot compete with coordinate descent for lasso problems.

**Bottom line**: This is a solid, correct implementation suitable for production use on small-to-medium problems. For performance parity with glmnet, coordinate descent (Strategy 1) is required.
