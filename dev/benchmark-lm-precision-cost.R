# Compare the optional precision diagnostic with a complete formula fit.
source_tree <- Sys.getenv("RMLX_AUDIT_SOURCE")
if (nzchar(source_tree)) {
  suppressPackageStartupMessages(devtools::load_all(
    source_tree, quiet = TRUE, compile = FALSE
  ))
} else {
  suppressPackageStartupMessages(library(Rmlx))
}
suppressPackageStartupMessages(devtools::load_all(quiet = TRUE))
iterations <- as.integer(Sys.getenv("RMLX_PRECISION_ITERATIONS", "30"))
out_dir <- Sys.getenv("RMLX_PRECISION_OUTPUT",
                      "dev/lm-precision-benchmark/diagnostic-cost")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

force_formula_fit <- function(data) {
  fit <- mlxs_lm(y ~ ., data = data)
  lapply(fit[c("coefficients", "fitted.values", "residuals")],
         Rmlx::mlx_eval)
  fit
}

specs <- data.frame(
  scenario = c("small", "narrow", "million_narrow", "wide"),
  n = c(1000L, 200000L, 1000000L, 60000L),
  p = c(6L, 51L, 51L, 201L)
)
timings <- list()
for (i in seq_len(nrow(specs))) {
  spec <- specs[i, ]
  set.seed(20261011)
  x <- matrix(rnorm(spec$n * (spec$p - 1L)), spec$n, spec$p - 1L)
  y <- 0.5 + drop(x %*% seq(-0.5, 0.5, length.out = spec$p - 1L)) +
    rnorm(spec$n)
  data <- data.frame(y, x)
  rm(x, y)
  fit <- force_formula_fit(data)
  score <- mlxs_lm_precision(fit, tol = Inf)
  qr_method <- fit$qr$method
  if (is.null(qr_method)) qr_method <- "cpu"
  for (iteration in seq_len(iterations)) {
    for (method in sample(c("fit", "diagnostic"))) {
      elapsed <- system.time({
        if (method == "fit") {
          fit <- force_formula_fit(data)
        } else {
          score <- mlxs_lm_precision(fit, tol = Inf)
        }
      })[["elapsed"]]
      timings[[length(timings) + 1L]] <- data.frame(
        spec, qr_method, iteration, method, seconds = elapsed
      )
    }
    cat(spec$scenario, iteration, "/", iterations, "\n")
  }
  write.csv(do.call(rbind, timings), file.path(out_dir, "timings.csv"),
            row.names = FALSE)
  rm(data, fit)
  gc()
}
timings <- do.call(rbind, timings)
summary <- aggregate(seconds ~ scenario + n + p + qr_method + method,
                     timings, function(x) c(median = median(x),
                                             p10 = quantile(x, 0.1),
                                             p90 = quantile(x, 0.9)))
summary <- cbind(summary[1:5], as.data.frame(summary$seconds))
names(summary)[6:8] <- c("median", "p10", "p90")
write.csv(summary, file.path(out_dir, "summary.csv"), row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(out_dir, "session.txt"))
print(summary, row.names = FALSE)
