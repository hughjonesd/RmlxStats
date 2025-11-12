# mlxs_glmnet Performance Analysis and Optimization Strategies

## Executive Summary

Current `mlxs_glmnet` is 10-154x slower than `glmnet::glmnet()`, with the gap narrowing on larger problems. The main bottlenecks are:

1. **Algorithm choice**: Proximal gradient descent vs coordinate descent
2. **Convergence speed**: More iterations needed per lambda
3. **Memory transfers**: Frequent R ↔ MLX conversions
4. **Limited GPU parallelism**: Sequential coordinate updates

## Current Implementation Analysis

### What glmnet Does

- **Algorithm**: Cyclical coordinate descent with soft thresholding
- **Pathwise optimization**: Warm starts from previous lambda solutions
- **Strong rules**: Pre-screening to reduce active set
- **Implementation**: Highly optimized Fortran
- **Speed**: 0.003-0.027s for test problems

### What mlxs_glmnet Does

- **Algorithm**: Proximal gradient descent with active sets
- **Pathwise optimization**: ✓ (has warm starts and strong rules)
- **Implementation**: R + Rmlx (MLX tensors)
- **Speed**: 0.271-0.464s for test problems
- **Bottlenecks**:
  - Lines 144-180: Inner loop with 4-6 MLX↔R conversions per iteration
  - Lines 154, 171, 177, 180: `as.numeric()` conversions
  - Lines 210-220: Soft threshold creates 7 temporary MLX objects
  - Line 62-64: Initial standardization

### Profiling Results

```
Problem Size     glmnet    mlxs_glmnet   Slowdown
n=500,  p=50    0.003s     0.464s        154.7x
n=2000, p=100   0.006s     0.253s        42.2x
n=5000, p=200   0.027s     0.271s        10.0x
```

Individual operations (per call):
- Matrix multiply: 0.24ms
- Crossprod: 0.26ms
- Soft threshold: 0.42ms
- Link function: 0.01ms
- MLX→R conversion: 0.04ms

The operations are fast, but with ~400,000 conversions for nlambda=100, maxit=1000, overhead adds up.

## Optimization Strategies

### Strategy 1: Batched Coordinate Descent (RADICAL)

**Concept**: Implement coordinate descent on GPU using parallel batched updates.

**Approach**:
- Update multiple coordinates simultaneously in parallel
- Use asynchronous/randomized coordinate selection (research: TPA-SCD algorithm)
- Batch process multiple lambda values at once
- Keep ALL computations in MLX tensors until final result

**Advantages**:
- Matches glmnet's proven algorithm
- Natural GPU parallelism (update p coordinates in parallel)
- Minimal R↔MLX transfers (only at start/end)
- Can process multiple lambdas simultaneously

**Challenges**:
- Coordinate descent is inherently sequential
- Need careful handling of shared memory
- Requires asynchronous updates (may sacrifice some convergence guarantees)
- More complex implementation

**Research Support**:
- "TPA-SCD can train SVM on tera-scale dataset in 1 minute on 4 GPUs"
- Parallel coordinate descent proven for separable convex functions
- Recent GPU implementations show 10-100x speedups

**Implementation Path**:
1. Implement basic coordinate descent in pure MLX (no R conversions in loop)
2. Add parallelization across coordinates (batched updates)
3. Add lambda-path batching (compute multiple lambdas together)
4. Optimize memory access patterns

### Strategy 2: Optimized Proximal Gradient (INCREMENTAL)

**Concept**: Keep current algorithm but eliminate bottlenecks.

**Approach**:
- Remove all inner-loop R↔MLX conversions
- Use MLX-native convergence checks
- Pre-compile the inner loop with `mlx_compile()`
- Batch compute multiple lambda values
- Simplify soft threshold to use fewer temporaries

**Advantages**:
- Lower risk (incremental improvements)
- Proven convergence properties
- Easier to maintain compatibility

**Challenges**:
- Still fundamentally slower convergence than coordinate descent
- May hit ceiling around 2-5x improvement

**Implementation Path**:
1. Rewrite inner loop to stay in MLX (lines 140-175)
2. Use MLX-native max/convergence operations
3. Compile hot paths with `mlx_compile()`
4. Profile and iterate

### Strategy 3: Hybrid Coordinate-Proximal (BALANCED)

**Concept**: Combine best of both worlds.

**Approach**:
- Use coordinate descent for the smooth part
- Use proximal operator for the penalty
- Implement as: θ_{j}^{k+1} = S_{λα}(θ_j^k - s∇_j f(θ^k))
- Keep computation in MLX, update coordinates in blocks

**Advantages**:
- Fast convergence of coordinate descent
- Simple proximal operator for L1 penalty
- Can parallelize within blocks
- Proven to work well for elastic net

**Challenges**:
- More complex than pure approaches
- Need to tune block size

**Implementation Path**:
1. Implement block coordinate descent structure
2. Integrate proximal gradient steps within blocks
3. Optimize block size for GPU memory
4. Add strong rules and active set management

### Strategy 4: Direct GPU Solver (ALTERNATIVE)

**Concept**: Use MLX's built-in optimization tools.

**Approach**:
- Formulate as differentiable objective (smooth approximation to L1)
- Use MLX's gradient computation and optimizers
- Implement proximal operator as custom MLX operation
- Leverage MLX's compilation and optimization

**Advantages**:
- Leverage MLX's optimized infrastructure
- Automatic differentiation
- Potentially good compiler optimizations

**Challenges**:
- L1 penalty non-differentiable (need smoothing or subgradient)
- May not match glmnet's path-following behavior
- Less control over convergence

### Strategy 5: Specialized Strong Rules (QUICK WIN)

**Concept**: Improve active set selection to reduce computation.

**Approach**:
- More aggressive strong rules screening
- Dynamic active set sizing based on GPU capacity
- Early stopping when active set stabilizes
- KKT violation checks less frequently

**Advantages**:
- Can combine with any other strategy
- Low implementation risk
- Proven effective in glmnet

**Expected Impact**: 20-40% speedup

## Recommended Approach

### Phase 1: Quick Wins (1-2 weeks)
- Implement Strategy 5 (better strong rules)
- Remove inner-loop conversions (Strategy 2, part 1)
- Expected: 2-3x speedup

### Phase 2: Algorithm Change (2-4 weeks)
- Implement Strategy 1 (batched coordinate descent) OR Strategy 3 (hybrid)
- Start with single-threaded coordinate descent in pure MLX
- Add parallelization incrementally
- Expected: 5-20x speedup (approaching glmnet speed)

### Phase 3: Advanced Optimization (if needed)
- Multi-lambda batching
- Compiled hot paths
- Optimized memory layout
- Expected: Additional 2-5x speedup

## GPU-Specific Considerations

### What GPUs Do Well
- Parallel matrix operations
- Batch processing
- Vectorized operations
- High throughput when saturated

### What GPUs Do Poorly
- Sequential operations
- Small operations with high overhead
- Frequent CPU↔GPU transfers
- Branching and conditionals

### Design Principles for GPU Implementation
1. **Batch everything**: Process multiple items together
2. **Stay on device**: Minimize CPU↔GPU transfers
3. **Saturate bandwidth**: Use all cores
4. **Coalesce memory**: Access contiguous memory
5. **Avoid synchronization**: Prefer asynchronous where safe

## Alternative Radical Ideas

### Idea 1: Multi-Response Batching
- Fit multiple Y vectors simultaneously
- Amortize setup costs
- GPU naturally parallel across responses
- Useful for bootstrap, cross-validation

### Idea 2: Approximate Path
- Don't compute all 100 lambdas
- Compute sparse grid (e.g., 10 values)
- Interpolate intermediate solutions
- Trade accuracy for speed

### Idea 3: Mixed Precision
- Use float16 for intermediate computations
- Keep float32 for final results
- 2x memory bandwidth improvement
- Acceptable for many applications

### Idea 4: Neural Network Approximation
- Train small NN to predict glmnet solution
- Use as warm start for iterative refinement
- Extremely fast inference
- Probably overkill, but interesting

## Conclusion

The most promising path forward is **Strategy 1 (Batched Coordinate Descent)** because:

1. Matches glmnet's proven algorithm
2. Natural fit for GPU parallelism
3. Research supports feasibility
4. Potential for 10-100x speedup

Combined with **Strategy 5 (improved strong rules)** as a quick win.

The implementation should:
- Keep ALL operations in MLX until final results
- Process coordinates in parallel batches
- Handle multiple lambda values simultaneously
- Use asynchronous updates where convergence permits

This is a significant rewrite, but necessary to achieve competitive performance with glmnet on GPU hardware.
