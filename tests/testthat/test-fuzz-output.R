test_that("fuzz metric rows use the canonical long schema", {
  rows <- fuzz_metric_rows(
    list(
      case_type = "monte_carlo",
      scenario = "schema_check",
      family = "gaussian",
      n = 20L,
      p = 3L,
      nreps = 10L
    ),
    term = c("x1", "x1"),
    measure = c("bias", "coverage"),
    target = c("coefficient", "confidence_interval"),
    source = "mlx",
    baseline = "truth",
    aggregation = "mean",
    value = c(0.02, 0.94),
    value_se = c(0.01, 0.02)
  )

  expect_named(rows, setdiff(fuzz_long_columns(), c(
    "branch", "commit_hash", "datetime_utc", "tier", "suite"
  )))
  expect_equal(rows$value, c(0.02, 0.94))
  expect_equal(rows$value_se, c(0.01, 0.02))
  expect_equal(rows$term, c("x1", "x1"))
  expect_true(all(is.na(rows$alpha)))
})

test_that("fuzz summary writer appends long rows", {
  out_dir <- "tmp-fuzz-output-test"
  out_path <- testthat::test_path(out_dir)
  withr::defer(unlink(out_path, recursive = TRUE))
  withr::local_envvar(RMLXSTATS_FUZZ_OUT = out_dir)

  rows <- fuzz_metric_rows(
    list(case_type = "deterministic", scenario = "writer_check", n = 10L),
    measure = "error",
    target = "coefficient",
    baseline = "reference",
    aggregation = "max",
    value = 1e-6
  )

  path <- write_fuzz_summaries(rows, suite = "schema", tier = "fast")
  path <- write_fuzz_summaries(rows, suite = "schema", tier = "fast")
  written <- utils::read.csv(path, check.names = FALSE)

  expect_equal(nrow(written), 2L)
  expect_named(written, fuzz_long_columns())
  expect_equal(written$measure, c("error", "error"))
})
