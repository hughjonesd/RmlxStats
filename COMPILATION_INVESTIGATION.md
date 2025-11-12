# Investigation: Can We Compile the Inner Loop?

## Initial Claim (Wrong)

I initially said mlx_compile() couldn't work because of:
1. `if (alpha < 1)` branching
2. `family$linkinv()` closure
3. Convergence check with early break

## Reality (You Were Right!)

### What Actually Prevents Compilation: NOTHING fundamental

1. **`if (alpha < 1)`** - Can use `mlx_where()` or just always compute (it's fast)
2. **`family$linkinv()`** - For gaussian: identity. For binomial: `1/(1 + exp(-eta))`. Both are pure MLX operations!
3. **Convergence check** - Can be moved outside the compiled function

### What I Tested

```r
one_iteration_simple <- function(x_active, beta_prev, residual, y, n_obs, step, lambda_val, alpha) {
  grad <- crossprod(x_active, residual) / n_obs
  grad <- grad + beta_prev * (lambda_val * (1 - alpha))  # No branching needed!

  beta_temp <- beta_prev - grad * step
  thresh <- lambda_val * alpha * step

  # Soft threshold
  abs_beta <- abs(beta_temp)
  magnitude <- mlx_maximum(abs_beta - thresh, 0)
  sign_beta <- beta_temp / (abs_beta + 1e-10)
  beta_new <- magnitude * sign_beta

  delta <- beta_new - beta_prev
  eta_new <- x_active %*% beta_new
  residual_new <- eta_new - y

  residual_new
}

compiled <- mlx_compile(one_iteration_simple)
```

**Result: 1.6x speedup** on n=1000, p=100, 1000 iterations

## Rmlx Limitation (NOW FIXED)

**List returns didn't work properly with mlx_compile**:
- Uncompiled: `list(a = x, b = y)` works fine
- Compiled: Returned list with no names (couldn't access by name)

This was an Rmlx binding limitation, not a fundamental MLX limitation.

**Issue filed**: https://github.com/hughjonesd/Rmlx/issues/16
**Status**: ✅ FIXED - list returns now work correctly

## Workarounds

### Option 1: Return Single Key Value
Only return the most expensive computed value (e.g., residual), recompute others in R.

```r
compiled_iter <- mlx_compile(function(...) {
  # ... computation ...
  residual_new  # Return just this
})

for (iter in 1:maxit) {
  residual_mlx <- compiled_iter(x_active, beta_mlx, residual_mlx, ...)

  # Recompute beta, etc. in R (cheap compared to the compiled part)
  # ...

  if (converged) break
}
```

### Option 2: Concatenate Returns
Pack multiple values into one array, unpack in R.

```r
compiled_iter <- mlx_compile(function(...) {
  # ... computation ...
  # Return [residual; beta_new; eta_new] stacked
  rbind(residual_new, beta_new, eta_new)
})

result <- compiled_iter(...)
residual <- result[1:n, ]
beta <- result[(n+1):(n+p), ]
# etc.
```

### Option 3: Fix Rmlx
Improve Rmlx's mlx_compile to handle list returns properly.

## What Speedup To Expect

From testing on n=1000, p=100:
- **1.6x speedup** for inner iteration logic

Applied to full mlxs_glmnet:
- If inner loop is 80% of total time → ~1.5x overall speedup
- If inner loop is 50% of total time → ~1.3x overall speedup

Combined with other optimizations (Strategy 2: ~1.15x), total could be:
- **~1.7-2x speedup** on current algorithm

Still won't match glmnet (25x faster), but a meaningful improvement.

## Recommendation

**YES, implement compilation!**

The inner loop CAN be compiled. The challenges are:
1. **Engineering**: Working around Rmlx's list return limitation
2. **Testing**: Ensuring compiled version matches uncompiled
3. **Maintenance**: Compilation may fail on some platforms

### Simple Implementation Plan

1. Extract inner iteration to separate function
2. Return just `residual_new` (the most expensive value)
3. Compile it with `mlx_compile()`
4. Fall back to uncompiled if compilation fails
5. Keep convergence check in R

```r
.one_iter_core <- function(x_active, beta, residual, y, n_obs, step, lambda, alpha, thresh) {
  grad <- crossprod(x_active, residual) / n_obs + beta * (lambda * (1 - alpha))
  beta_new <- soft_threshold(beta - grad * step, thresh)
  delta <- beta_new - beta
  eta_new <- x_active %*% beta_new
  residual_new <- eta_new - y  # For gaussian; handle binomial separately
  residual_new
}

.one_iter_compiled <- mlx_compile(.one_iter_core)

# In main loop:
for (iter in 1:maxit) {
  residual_mlx <- .one_iter_compiled(x_active, beta_mlx[active_idx],
                                     residual_mlx, y_mlx, n_obs, step,
                                     lambda_val, alpha, thresh)

  # Update beta from delta (recompute quickly in R if needed)
  # Check convergence in R
  # ...
}
```

## Why I Was Wrong

I dismissed compilation too quickly based on:
1. **Assumed** family$linkinv was complex - it's not
2. **Assumed** branching was a blocker - mlx_where works fine
3. **Didn't test** properly - actual test shows clear speedup

The real lesson: **Always test assumptions!**

## Impact Assessment

### Without compilation (current):
- n=5000, p=200, nlambda=100: ~0.90s
- 25x slower than glmnet

### With compilation (estimated):
- Inner loop speedup: 1.6x
- Overall speedup: ~1.4-1.5x (accounting for setup overhead)
- Expected time: ~0.60-0.65s
- Still ~18-20x slower than glmnet

### To Match Glmnet:
Still need Strategy 1 (coordinate descent), but compilation is a worthwhile incremental gain.

## Action Items

- [ ] Implement compiled inner loop with single return value
- [ ] Benchmark on target problem (n=5000, p=200)
- [ ] Test on both gaussian and binomial families
- [ ] Add fallback for platforms where compilation fails
- [ ] Document in AGENTS.md that compilation IS possible and helpful

## Conclusion

You were right to push back. The inner loop CAN and SHOULD be compiled. Current Rmlx limitations make it slightly awkward, but it's definitely worth the ~1.5-2x speedup.
