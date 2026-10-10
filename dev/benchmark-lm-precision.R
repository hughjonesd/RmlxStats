# Run from the repository root with a merged Rmlx installation or source tree.
source_tree <- Sys.getenv("RMLX_AUDIT_SOURCE")
if (nzchar(source_tree)) {
  suppressPackageStartupMessages(devtools::load_all(
    source_tree, quiet = TRUE, compile = FALSE
  ))
} else {
  suppressPackageStartupMessages(library(Rmlx))
}
suppressPackageStartupMessages(devtools::load_all(
  Sys.getenv("RMLX_STATS_SOURCE", "."), quiet = TRUE
))
iterations <- as.integer(Sys.getenv("RMLX_PRECISION_ITERATIONS", "30"))
out_dir <- Sys.getenv("RMLX_PRECISION_OUTPUT", "dev/lm-precision-benchmark/run")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
writeLines(capture.output(sessionInfo()), file.path(out_dir, "session.txt"))

force_fit <- function(method, x32, y32, x64, y64) {
  fit <- if (method == "float64") {
    Rmlx::with_device("cpu", mlxs_lm_fit(x64, y64, qr_method = "cpu"))
  } else {
    mlxs_lm_fit(x32, y32, qr_method = if (method == "cpu32") "cpu" else {
      "auto"
    })
  }
  lapply(fit[c("coefficients", "fitted.values", "residuals")],
         Rmlx::mlx_eval)
  fit
}

specs <- data.frame(
  scenario = c("iid", "low_noise_correlated", "million_iid"),
  n = c(200000L, 200000L, 1000000L), p = 51L,
  condition = c(1, 100, 1), noise_sd = c(1, 1e-4, 1)
)
methods <- strsplit(Sys.getenv(
  "RMLX_PRECISION_METHODS", "current,cpu32,float64"
), ",")[[1L]]
timings <- list()
metrics <- list()
for (i in seq_len(nrow(specs))) {
  spec <- specs[i, ]
  set.seed(20261011)
  x <- cbind(1, matrix(rnorm(spec$n * (spec$p - 1L)),
                       spec$n, spec$p - 1L))
  if (spec$condition > 1) {
    x[, 3L] <- x[, 2L] + 2 / spec$condition * x[, 3L]
  }
  y <- drop(x %*% seq(-0.5, 0.5, length.out = spec$p)) +
    rnorm(spec$n, sd = spec$noise_sd)
  ref <- lm.fit(x, y)
  se <- sqrt(diag(base::chol2inv(qr.R(ref$qr))) *
               sum(ref$residuals^2) / (spec$n - spec$p))
  x32 <- Rmlx::as_mlx(x)
  y32 <- Rmlx::mlx_matrix(y, ncol = 1L)
  x64 <- Rmlx::as_mlx(x, dtype = "float64")
  y64 <- Rmlx::mlx_matrix(y, ncol = 1L, dtype = "float64")
  lapply(list(x32, y32, x64, y64), Rmlx::mlx_eval)
  for (method in methods) {
    fit <- force_fit(method, x32, y32, x64, y64)
    coef <- drop(as.matrix(fit$coefficients))
    metrics[[length(metrics) + 1L]] <- data.frame(
      spec, method, coef_max_abs = max(abs(coef - ref$coefficients)),
      coef_error_se = max(abs(coef - ref$coefficients) / se)
    )
  }
  # Randomize each round to distribute warm-up and thermal drift.
  for (iteration in seq_len(iterations)) {
    for (method in sample(methods)) {
      elapsed <- system.time({
        fit <- force_fit(method, x32, y32, x64, y64)
      })[["elapsed"]]
      timings[[length(timings) + 1L]] <- data.frame(
        scenario = spec$scenario, iteration, method, seconds = elapsed
      )
    }
    write.csv(do.call(rbind, timings), file.path(out_dir, "timings.csv"),
              row.names = FALSE)
    cat(spec$scenario, iteration, "/", iterations, "\n")
  }
}
metrics <- do.call(rbind, metrics)
timings <- do.call(rbind, timings)
write.csv(metrics, file.path(out_dir, "precision.csv"), row.names = FALSE)
summary <- aggregate(seconds ~ scenario + method, timings,
                     function(x) c(median = median(x),
                                    p10 = quantile(x, 0.1),
                                    p90 = quantile(x, 0.9)))
summary <- cbind(summary[1:2], as.data.frame(summary$seconds))
names(summary)[3:5] <- c("median", "p10", "p90")
write.csv(summary, file.path(out_dir, "summary.csv"), row.names = FALSE)
print(summary, row.names = FALSE)
