# Repository Guidelines

This project builds on the `Rmlx` GPU math package to expose statistical workflows.
Many conventions mirror the upstream repository; deviations and RmlxStats-specific
guidance are called out below.

## Project Structure & Module Organization
- `R/` holds exported R wrappers, S3 methods, and roxygen docs. Mirror upstream
  layout like `mlxs-lm.R` when adding public APIs (use the `mlxs_` prefix).
- `tests/testthat/` groups unit specs by domain (`test-mlxs-lm.R`, `test-pca.R`);
  add new files as `test-feature.R`.
- Vignettes (planned) will live in `vignettes/`; update when user-facing features land.
- `DESCRIPTION`, `NAMESPACE`, and `AGENTS.md` manage package metadata and coordination docs.

## Build, Test, and Development Commands
- `R -q -e 'devtools::document()'` rebuilds NAMESPACE and Rd files from roxygen comments.
- `R -q -e 'devtools::build()'` creates a source tarball; `R -q -e 'devtools::check()'`
  runs formal package checks.
- `R -q -e 'devtools::test()'` runs the testthat suite; use
  `R -q -e 'devtools::load_all()'` for rapid iteration.
- When Rcpp glue is added, also run `R -q -e 'Rcpp::compileAttributes()'`
  to regenerate `RcppExports`.

## Coding Style & Naming Conventions
- Use two-space indents in R; keep lines under 100 characters to match upstream style.
- Public functions use the `mlxs_` prefix (`mlxs_lm`, `mlxs_glm`). S3 methods follow
  `generic.class` (`print.mlxs_lm`).
- Document R functions with roxygen `#'` blocks; let `@export` drive NAMESPACE entries.
- Prefer snake_case for internal helpers (`as_model_matrix`).

## Testing Guidelines
- Write tests with testthat in `tests/testthat`; keep scenarios focused and readable.
- Use CPU-friendly fixtures (small matrices) so GPU and CPU paths run quickly.
- Run `R -q -e 'devtools::test()'` locally; on other machines failures are acceptable when MLX
  is absent, but prefer explicit skips with informative messages if GPU is required.
- Within this workspace you can assume MLX and Rmlx are installed and working; do not
  add `skip_if_not_available` scaffolding around MLX usage unless directed otherwise.

## Working Style Expectations
- **KEEP IT SIMPLE.** Prefer the most direct expression of an idea over elaborate helper
  stacks. If a single call (e.g., `as_mlx()`) communicates intent, use it instead of
  wrapping the same logic in multiple conditionals.
- **READ THE MLX AND LOCAL CODEBASE.** Before changing behaviour, scan the existing MLX
  helpers and this repo to stay aligned with current conventions—assume the answer
  probably already exists somewhere nearby.

## Commit & Pull Request Guidelines
- Follow imperative, capitalized commit messages (e.g., `Add mlxs_lm interface`).
- Document API changes and summarize GPU/CPU paths covered in PR descriptions.
- Before opening a PR, run `document()`, `test()`, and `check()`; include notable
  performance notes if GPU speedups are claimed.

## Current Agent Notes (2025-10-31)
- MLX tensor creation and Metal-backed work only succeeds with a permissive sandbox
  (e.g., `danger-full-access`). Restricted sandboxes may block Metal device
  initialisation and `processx`/`callr` usage.
- Under restricted modes you can still run `document()`, `test()` (CPU paths), and
  `check()`, but GPU calls may throw `c++ exception (unknown reason)`.
- Keep the workspace tidy after checks: remove temporary `*.tar.gz` artifacts or
  `.Rcheck/` directories created during manual workflows.

## Additional Guidance

### MLX-first Data Handling
- This package exists to validate an *mlx*-based statistics workflow. Move inputs into
  MLX arrays immediately and keep them there as long as feasible; prefer MLX outputs
  for anything that could feed other computations.
- Base R representations are acceptable only for user-facing summaries (printing,
  glance/tidy outputs, etc.) or when required by R generics.
- When constructing MLX data from raw R vectors/scalars, favor explicit constructors
  such as `Rmlx::mlx_scalar()` / `Rmlx::mlx_vector()` / `Rmlx::mlx_array()` rather
  than `as_mlx()` so future readers can see intent at the call site.

### Issue Tracking
- Log backlog ideas directly as GitHub issues (use `gh issue create ...`) instead
  of keeping a local `docs/github-issues.md` scratchpad. Include the issue number
  in PR summaries for traceability.
- Residual bootstraps for `mlxs_glm` are only supported for (quasi)gaussian
  families—fail fast if anything else is requested rather than silently
  downgrading to case resampling.
- Keep bootstrap implementations MLX-native end-to-end: gather samples via MLX
  subsetting, refit with `mlxs_lm_fit` / `.mlxs_glm_fit_core`, and return MLX
  standard-error columns (no intermediate `as.matrix()`/`stats::sd` hops).
- When mutating MLX arrays via `[<-`, convert the update to a base matrix first
  if needed to avoid `as.vector.mlx` warnings (e.g., large active-set updates
  in `mlxs_glmnet`).

### Integration with Rmlx
- Always import Rmlx helpers (`as_mlx`, `mlx_matmul`, `qr.mlx`, etc.) via the
  qualified namespace (`Rmlx::`). Update `DESCRIPTION` Imports/Remotes as needed.
- Confirm function names against the latest pkgdown docs:
  https://hughjonesd.github.io/Rmlx/reference/index.html.
- R arrays are column-major while MLX tensors are row-major; double-check axis
  handling if reductions behave unexpectedly.
- Use explicit `Rmlx::colMeans()` and `Rmlx::colSums()` rather than unqualified
  calls to avoid namespace confusion between base R and Rmlx methods.

### Testing Notes
- Tests should compare MLX-backed results to base-R equivalents (e.g., `stats::lm`).
- Use `expect_equal(..., tolerance = 1e-6)` when asserting floating point equality.
- To iterate on a single spec: `R -q -e 'devtools::test_file("tests/testthat/test-mlxs-lm.R")'`.

### Documentation Workflow
- Roxygen comments power `man/` docs; regenerate with `devtools::document()`
  after editing.
- Favor markdown lists/tables in roxygen over `\item`.
- Update the pkgdown site configuration (future work) so new exports appear in the
  reference index.

### MLX Performance Characteristics (Learned from mlxs_glmnet optimization)

#### What MLX Does Well
- **Parallel matrix operations**: Large matrix multiplications, crossprod, etc.
- **Batch processing**: Operating on multiple data elements simultaneously
- **Staying on device**: Keeping data in MLX throughout a computation pipeline
- **Vectorized operations**: Element-wise operations on large arrays

#### What MLX Does Poorly (vs Fortran/C++)
- **Small sequential operations**: Building computation graphs has overhead
- **Frequent MLX↔R conversions**: Each conversion triggers evaluation
- **Branching logic**: Conditionals and early exits in tight loops
- **Single-element comparisons**: `as.logical(x > tol)` for scalars has overhead

#### Conversion Overhead Is Usually Small
Investigation of `mlxs_glmnet` revealed:
- Eliminated ~20,000-50,000 conversions per run
- Performance impact: ~0% (within measurement noise)
- **Algorithm choice matters far more than conversion count**

#### When to Convert to MLX
- **Early conversion wins**: Convert inputs to MLX immediately after validation
- **Late conversion to R**: Only convert final results back to R
- **Keep intermediate results in MLX**: Even if you need one value in R, keep the
  MLX version around for subsequent computations

#### MLX Axis Handling
- MLX uses C-style axis numbering in underlying C++, but Rmlx adapts to R conventions
- Test axis operations carefully; sometimes best to compute in R then convert result

#### Comparison Operators in Conditionals
- MLX comparison returns MLX array: `x > tol` returns mlx object
- Cannot use directly in `if()`: must convert with `as.logical()` or `as.numeric()`
- Example: `if (as.logical(delta < tol))` works; `if (delta < tol)` errors

#### mlx_compile() Capabilities and Limitations

**What CAN be compiled:**
- Pure MLX operations: matrix multiply, element-wise ops, reductions
- Conditional logic using `mlx_where()` instead of `if/else`
- Simple mathematical expressions (`1/(1 + exp(-x))`, etc.)
- Functions returning a single MLX array

**Measured speedups:**
- Simple iteration logic: 1.5-1.6x speedup
- Complex compiled functions: potentially higher
- Worth implementing for hot paths

**When to use:**
- Inner loops executing 100s-1000s of times
- Pure computational logic without side effects
- When profiling shows the function is a bottleneck

**Example pattern:**
```r
inner_core <- function(x, y, params) {
  # Pure MLX computation
  result <- ... # expensive operations
  result
}

inner_compiled <- mlx_compile(inner_core)

for (iter in 1:max_iter) {
  result <- inner_compiled(x, y, params)
  # Convergence check in R
  if (converged(result)) break
}
```

#### Algorithm Design for GPU
- **Think parallel**: Can multiple coordinates/lambdas be processed simultaneously?
- **Batch operations**: Group similar operations together
- **Avoid sequential updates**: Each sequential step wastes GPU parallelism
- **Example**: Coordinate descent naturally parallelizes across coordinates;
  proximal gradient is inherently sequential

#### Storage in MLX
- Keep result storage arrays (`beta_store`, `intercept_store`) as MLX
- Assign directly to MLX arrays: `beta_store_mlx[, idx] <- beta_mlx`
- Only convert entire result array at the end
- Eliminates per-iteration conversion overhead (though often small)

#### When GPU Won't Help
The `mlxs_glmnet` optimization showed GPU provides no speedup when:
1. Algorithm is inherently sequential (e.g., proximal gradient descent)
2. Operations are small and frequent (graph-building overhead dominates)
3. No opportunity for parallel processing
4. Highly optimized CPU code exists (e.g., Fortran glmnet)

In such cases, focus on algorithmic improvements over GPU utilization.

### Handy Tips
- `usethis::` helpers streamline chores; prefer them for package setup.
- MLX arrays currently lack a default constructor; supply shape/dtype explicitly
  when wiring future C++ glue.
- Discover available helpers with `library(help = "Rmlx")` or
  `ls(envir = asNamespace("Rmlx"), all.names = TRUE)`.
- In user-facing docs, prefer the term *array* over *tensor* to match R conventions.
- Add concise internal comments for non-obvious helper logic; keep the codebase
  self-explanatory.
- Use `mlx_zeros()` to pre-allocate result arrays rather than converting from R
- Profile before optimizing: conversion overhead is often smaller than expected
