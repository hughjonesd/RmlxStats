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

### Handy Tips
- `usethis::` helpers streamline chores; prefer them for package setup.
- MLX arrays currently lack a default constructor; supply shape/dtype explicitly
  when wiring future C++ glue.
- Discover available helpers with `library(help = "Rmlx")` or
  `ls(envir = asNamespace("Rmlx"), all.names = TRUE)`.
- In user-facing docs, prefer the term *array* over *tensor* to match R conventions.
- Add concise internal comments for non-obvious helper logic; keep the codebase
  self-explanatory.
