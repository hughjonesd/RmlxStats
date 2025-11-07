# Repository Guidelines

This project builds on the `Rmlx` GPU math package to expose statistical
workflows. Many conventions mirror the upstream repository; deviations
and RmlxStats-specific guidance are called out below.

## Project Structure & Module Organization

- `R/` holds exported R wrappers, S3 methods, and roxygen docs. Mirror
  upstream layout like `mlxs-lm.R` when adding public APIs (use the
  `mlxs_` prefix).
- `tests/testthat/` groups unit specs by domain (`test-mlxs-lm.R`,
  `test-pca.R`); add new files as `test-feature.R`.
- Vignettes (planned) will live in `vignettes/`; update when user-facing
  features land.
- `DESCRIPTION`, `NAMESPACE`, and `AGENTS.md` manage package metadata
  and coordination docs.

## Build, Test, and Development Commands

- `R -q -e 'devtools::document()'` rebuilds NAMESPACE and Rd files from
  roxygen comments.
- `R -q -e 'devtools::build()'` creates a source tarball;
  `R -q -e 'devtools::check()'` runs formal package checks.
- `R -q -e 'devtools::test()'` runs the testthat suite; use
  `R -q -e 'devtools::load_all()'` for rapid iteration.
- When Rcpp glue is added, also run
  `R -q -e 'Rcpp::compileAttributes()'` to regenerate `RcppExports`.

## Coding Style & Naming Conventions

- Use two-space indents in R; keep lines under 100 characters to match
  upstream style.
- Public functions use the `mlxs_` prefix (`mlxs_lm`, `mlxs_glm`). S3
  methods follow `generic.class` (`print.mlxs_lm`).
- Document R functions with roxygen `#'` blocks; let `@export` drive
  NAMESPACE entries.
- Prefer snake_case for internal helpers (`as_model_matrix`).

## Testing Guidelines

- Write tests with testthat in `tests/testthat`; keep scenarios focused
  and readable.
- Use CPU-friendly fixtures (small matrices) so GPU and CPU paths run
  quickly.
- Run `R -q -e 'devtools::test()'` locally; failures are acceptable when
  MLX is absent, but prefer explicit skips with informative messages if
  GPU is required.

## Commit & Pull Request Guidelines

- Follow imperative, capitalized commit messages (e.g.,
  `Add mlxs_lm interface`).
- Document API changes and summarize GPU/CPU paths covered in PR
  descriptions.
- Before opening a PR, run `document()`, `test()`, and `check()`;
  include notable performance notes if GPU speedups are claimed.

## Current Agent Notes (2025-10-31)

- MLX tensor creation and Metal-backed work only succeeds with a
  permissive sandbox (e.g., `danger-full-access`). Restricted sandboxes
  may block Metal device initialisation and `processx`/`callr` usage.
- Under restricted modes you can still run `document()`, `test()` (CPU
  paths), and `check()`, but GPU calls may throw
  `c++ exception (unknown reason)`.
- Keep the workspace tidy after checks: remove temporary `*.tar.gz`
  artifacts or `.Rcheck/` directories created during manual workflows.

## Additional Guidance

### Integration with Rmlx

- Always import Rmlx helpers (`as_mlx`, `mlx_matmul`, `qr.mlx`, etc.)
  via the qualified namespace (`Rmlx::`). Update `DESCRIPTION`
  Imports/Remotes as needed.
- Confirm function names against the latest pkgdown docs:
  <https://hughjonesd.github.io/Rmlx/reference/index.html>.
- R arrays are column-major while MLX tensors are row-major;
  double-check axis handling if reductions behave unexpectedly.

### Testing Notes

- Tests should compare MLX-backed results to base-R equivalents (e.g.,
  [`stats::lm`](https://rdrr.io/r/stats/lm.html)).
- Use `expect_equal(..., tolerance = 1e-6)` when asserting floating
  point equality.
- To iterate on a single spec:
  `R -q -e 'devtools::test_file("tests/testthat/test-mlxs-lm.R")'`.

### Documentation Workflow

- Roxygen comments power `man/` docs; regenerate with
  `devtools::document()` after editing.
- Favor markdown lists/tables in roxygen over `\item`.
- Update the pkgdown site configuration (future work) so new exports
  appear in the reference index.

### Handy Tips

- `usethis::` helpers streamline chores; prefer them for package setup.
- MLX arrays currently lack a default constructor; supply shape/dtype
  explicitly when wiring future C++ glue.
- Discover available helpers with
  [`library(help = "Rmlx")`](https://rdrr.io/r/base/library.html) or
  `ls(envir = asNamespace("Rmlx"), all.names = TRUE)`.
- In user-facing docs, prefer the term *array* over *tensor* to match R
  conventions.
- Add concise internal comments for non-obvious helper logic; keep the
  codebase self-explanatory.
