# Repository Guidelines

This project builds on the `Rmlx` GPU math package to expose statistical workflows.


## Project Structure & Module Organization
- `R/` holds R code and roxygen docs. Use the `mlxs_` prefix for public functions.
- `tests/testthat/` groups unit specs by domain (`test-mlxs-lm.R`, `test-pca.R`);
  add new files as `test-feature.R`.

## Build, Test, and Development Commands
- `R -q -e 'devtools::document()'` rebuilds NAMESPACE and Rd files from roxygen comments.
- `R -q -e 'devtools::build()'` creates a source tarball; 
- `R -q -e 'devtools::check()'` runs formal package checks. Use `vignette = FALSE`
  to avoid building slow vignettes in the check.
- `R -q -e 'devtools::test()'` runs tests. 
- Use `R -q -e 'devtools::load_all()'` for rapid iteration.
- When Rcpp glue is added, also run `R -q -e 'Rcpp::compileAttributes()'`
  to regenerate `RcppExports`.


## Development and Coding
- Use two-space indents in R; keep lines under 80 characters.
- Document R functions with roxygen `#'` blocks; let `@export` drive NAMESPACE entries.
  Use markdown formatting in roxygen.
- Prefer snake_case.
- After adding a function, document it with roxygen; add it to `_pkgdown.yml`;
  and add a line to `NEWS.md`.

## Testing
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
- **READ THE Rmlx AND LOCAL CODEBASE.** Before changing behaviour, scan the existing MLX
  helpers and this repo to stay aligned with current conventions—assume the answer
  probably already exists somewhere nearby. `Rmlx` is in `../Rmlx`. You should read
  `AGENTS.md` there too. If you find an issue in Rmlx, you can fix it yourself,
  or file a github issue using the `gh` command line tool.
- **CHECK YOUR CODE BEFORE THEORIZING.** When debugging performance issues, verify the
  implementation is correct before building theories about algorithmic trade-offs or
  architectural constraints. Profile to find bottlenecks, then inspect the actual code
  at those hot spots. Be empirical and modest about your capabilities.
- **KEEP VARIABLES IN MLX**. The package exists to test the `Rmlx` package. 
  Any speedups can 
  only be gained by using Rmlx data structures. Any data-dependent variables
  (broadly, anything on the scale of "n" or "p", including e.g. regression 
  coefficients) should be mlx variables. Mlx variables should be accepted as
  inputs (use `as_mlx()`). Computations should be done using Rmlx including
  mlx-specific methods for base generics (see `help("mlx-methods")`). If you
  have to convert to R, that is a failure: ask for help or rethink. Familiarize
  yourself with `mlx_matrix()`, `mlx_vector()` and `as_mlx()` for conversion; 
  look at `ls(package="Rmlx")` for mlx functions.
  
## Other Notes

### Testing Notes
- Tests should compare MLX-backed results to base-R equivalents (e.g., `stats::lm`).
- Use `expect_equal(..., tolerance = 1e-6)` when asserting floating point equality.
- To iterate on a single spec: `R -q -e 'devtools::test_file("tests/testthat/test-mlxs-lm.R")'`.


### Comparison Operators in Conditionals
- MLX comparison returns MLX array: `x > tol` returns mlx object
- Cannot use directly in `if()`: must convert with `as.logical()` or `as.numeric()`
- Example: `if (as.logical(delta < tol))` works; `if (delta < tol)` errors
- `any()` and `all()` return R logical vectors as normal


### Algorithm Design for GPU
- **Think parallel**: Can multiple coordinates/lambdas be processed simultaneously?
- **Batch operations**: Group similar operations together
- **Avoid sequential updates**: Each sequential step wastes GPU parallelism
- **Example**: Coordinate descent naturally parallelizes across coordinates;
  proximal gradient is inherently sequential


### Handy Tips
- `usethis::` helpers streamline chores; prefer them for package setup.
- MLX arrays currently lack a default constructor; supply shape/dtype explicitly
  when wiring future C++ glue.
- Discover available helpers with `library(help = "Rmlx")` or
  `ls(envir = asNamespace("Rmlx"), all.names = TRUE)`.
- In user-facing docs, prefer the term *array* over *tensor* to match R conventions.
- Add concise internal comments for non-obvious helper logic; keep the codebase
  self-explanatory.

### MLX Ops Cheat Sheet
- Base-R logical/comparison generics (`|`, `&`, `!`, `any()`, `all()`, `sum()`,
  `length()`, comparisons, etc.) already dispatch through MLX, so they operate
  entirely on-device until you explicitly call `as.logical()`/`as.numeric()`.
- Indexing accepts MLX masks or integer arrays directly: both
  `x[mlx_boolean_mask, ]` and `x[as_mlx(int_idx), ]` work without intermediate
  host copies.
- Sequence helpers don’t need R vectors first—use `seq.int(length(dim(x)))` or
  `Rmlx::mlx_arange()` to build MLX index ranges.
- Keep gradients, active-set masks, and other n/p-sized vectors as MLX arrays;
  drop to R only once when you truly need an integer index (e.g., `which()`).
