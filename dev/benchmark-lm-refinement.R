suppressPackageStartupMessages(devtools::load_all(quiet = TRUE))

force_fit <- function(fit) {
  as.matrix(fit$coefficients)
  as.matrix(fit$fitted.values)
  fit
}

make_case <- function(kind, n = 2e5, p = 50) {
  set.seed(match(kind, c("iid", "ar1", "near_collinear", "weighted")))
  x <- matrix(rnorm(n * p), nrow = n)
  if (kind == "ar1") {
    for (j in seq.int(2, p)) x[, j] <- 0.9 * x[, j - 1] + 0.1 * x[, j]
  }
  if (kind == "near_collinear") {
    x[, 2] <- x[, 1] + rnorm(n, sd = 1e-5)
  }
  x <- cbind(1, x)
  beta <- c(1, seq(0.75, by = -0.15, length.out = p))
  y <- drop(x %*% beta) + rnorm(n, sd = 0.01)
  weights <- if (kind == "weighted") seq(0.25, 1.5, length.out = n) else NULL
  list(x = x, y = y, weights = weights)
}

fit_metrics <- function(kind, epsilon_f64 = 1e-6) {
  case <- make_case(kind)
  x <- case$x
  y <- case$y
  weights <- case$weights
  ref <- if (is.null(weights)) {
    lm.fit(x, y, tol = 1e-16)
  } else {
    lm.wfit(x, y, w = weights, tol = 1e-16)
  }
  y_col <- Rmlx::mlx_matrix(y, ncol = 1)
  w_col <- if (is.null(weights)) NULL else Rmlx::mlx_matrix(weights, ncol = 1)

  t_f32 <- system.time({
    f32 <- force_fit(mlxs_lm_fit(Rmlx::as_mlx(x), y_col, weights = w_col))
  })[["elapsed"]]
  t_refined <- system.time({
    refined <- force_fit(mlxs_lm_fit(
      Rmlx::as_mlx(x), y_col, weights = w_col, epsilon_f64 = epsilon_f64
    ))
  })[["elapsed"]]
  t_f64 <- system.time({
    x64 <- Rmlx::mlx_cast(Rmlx::as_mlx(x), dtype = "float64", device = "cpu")
    y64 <- Rmlx::mlx_cast(y_col, dtype = "float64", device = "cpu")
    w64 <- if (is.null(w_col)) NULL else {
      Rmlx::mlx_cast(w_col, dtype = "float64", device = "cpu")
    }
    f64 <- force_fit(mlxs_lm_fit(x64, y64, weights = w64))
  })[["elapsed"]]

  err <- function(fit) {
    c(
      fitted_error = max(abs(drop(as.matrix(fit$fitted.values)) -
        ref$fitted.values)),
      coef_error = max(abs(drop(as.matrix(fit$coefficients)) -
        ref$coefficients)),
      rss_delta = sum(drop(as.matrix(fit$residuals))^2) - sum(ref$residuals^2)
    )
  }

  data.frame(
    case = kind,
    method = c("f32", "refined", "direct_f64"),
    kappa = kappa(x),
    rbind(err(f32), err(refined), err(f64)),
    time = c(t_f32, t_refined, t_f64),
    trigger = c(NA_real_, refined$refinement_initial_error, NA_real_),
    final = c(NA_real_, refined$refinement_final_error, NA_real_),
    iterations = c(0L, refined$refinement_iterations, 0L)
  )
}

cases <- c("iid", "ar1", "near_collinear", "weighted")
results <- do.call(rbind, lapply(cases, fit_metrics))
num_cols <- vapply(results, is.numeric, logical(1))
#results[num_cols] <- lapply(results[num_cols], signif, digits = 4)
#print(results, row.names = FALSE)
