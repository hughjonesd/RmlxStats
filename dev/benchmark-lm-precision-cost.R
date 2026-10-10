# Compare complete formula fits with and without the automatic final check.
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
                      "dev/lm-precision-benchmark/automatic-check-cost")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

force_formula_fit <- function(data, model, precision_tol) {
  fit <- if (model == "lm") {
    mlxs_lm(y ~ ., data = data, precision_tol = precision_tol)
  } else {
    family <- switch(model, gaussian = mlxs_gaussian(),
                      binomial = mlxs_binomial(), poisson = mlxs_poisson())
    mlxs_glm(y ~ ., data = data, family = family,
             precision_tol = precision_tol)
  }
  lapply(fit[c("coefficients", "fitted.values", "residuals")],
         Rmlx::mlx_eval)
  fit
}

specs <- data.frame(
  scenario = c("small", "narrow", "million_narrow", "wide",
               "glm_gaussian", "glm_binomial", "glm_poisson"),
  model = c(rep("lm", 4), "gaussian", "binomial", "poisson"),
  n = c(1000L, 200000L, 1000000L, 60000L, rep(200000L, 3)),
  p = c(6L, 51L, 51L, 201L, rep(51L, 3))
)
timings <- list()
for (i in seq_len(nrow(specs))) {
  spec <- specs[i, ]
  set.seed(20261011)
  x <- matrix(rnorm(spec$n * (spec$p - 1L)), spec$n, spec$p - 1L)
  eta <- 0.5 + drop(x %*% seq(-0.1, 0.1, length.out = spec$p - 1L))
  y <- switch(spec$model, lm = eta + rnorm(spec$n),
               gaussian = eta + rnorm(spec$n),
               binomial = rbinom(spec$n, 1, plogis(eta)),
               poisson = rpois(spec$n, exp(eta)))
  data <- data.frame(y, x)
  rm(x, y, eta)
  for (method in c("unchecked", "checked")) {
    fit <- force_formula_fit(data, spec$model,
                             if (method == "checked") 0.1 else NULL)
  }
  qr_method <- fit$qr$method
  if (is.null(qr_method)) qr_method <- "cpu"
  for (iteration in seq_len(iterations)) {
    for (method in sample(c("unchecked", "checked"))) {
      elapsed <- system.time({
        fit <- force_formula_fit(data, spec$model,
                                 if (method == "checked") 0.1 else NULL)
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
summary <- aggregate(seconds ~ scenario + model + n + p + qr_method + method,
                     timings, function(x) c(median = median(x),
                                             p10 = quantile(x, 0.1),
                                             p90 = quantile(x, 0.9)))
summary <- cbind(summary[1:6], as.data.frame(summary$seconds))
names(summary)[7:9] <- c("median", "p10", "p90")
write.csv(summary, file.path(out_dir, "summary.csv"), row.names = FALSE)
print(summary, row.names = FALSE)
