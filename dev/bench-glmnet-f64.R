devtools::load_all(quiet = TRUE)

loadNamespace("glmnet")

make_ar1 <- function(n, p, rho = 0.8) {
  x <- matrix(rnorm(n * p), nrow = n)
  for (j in seq.int(2L, p)) {
    x[, j] <- rho * x[, j - 1L] + sqrt(1 - rho^2) * x[, j]
  }
  x <- scale(x)
  matrix(as.numeric(x), nrow = n)
}

scenario_data <- function(name) {
  set.seed(match(name, c("dense_gaussian", "gram_gaussian", "binomial")))
  if (name == "dense_gaussian") {
    x <- make_ar1(250, 80, rho = 0.9)
    beta <- c(runif(8, -1, 1), rep(0, 72))
    return(list(x = x, y = drop(x %*% beta + rnorm(250)), family = "gaussian"))
  }
  if (name == "gram_gaussian") {
    x <- make_ar1(1200, 20, rho = 0.7)
    beta <- c(runif(5, -1, 1), rep(0, 15))
    return(list(x = x, y = drop(x %*% beta + rnorm(1200)), family = "gaussian"))
  }
  x <- make_ar1(300, 40, rho = 0.8)
  beta <- c(runif(6, -1, 1), rep(0, 34))
  prob <- plogis(drop(x %*% beta))
  list(x = x, y = rbinom(300, 1, prob), family = "binomial")
}

fit_once <- function(dat, method, lambda) {
  family <- if (dat$family == "gaussian") mlxs_gaussian() else mlxs_binomial()
  x <- dat$x
  y <- dat$y
  if (method == "full_f64") {
    Rmlx::local_device("cpu")
    x <- matrix(as.numeric(x), nrow = nrow(dat$x))
    x <- Rmlx::as_mlx(x, dtype = "float64")
    y <- Rmlx::as_mlx(as.numeric(y), dtype = "float64")
  }

  elapsed <- system.time({
    fit <- if (method == "old_f32") {
      mlxs_glmnet(
        x = x,
        y = y,
        family = family,
        lambda = lambda,
        standardize = TRUE,
        maxit = 1000,
        tol = 1e-6,
        tol_f64 = NULL
      )
    } else {
      mlxs_glmnet(
        x = x,
        y = y,
        family = family,
        lambda = lambda,
        standardize = TRUE,
        maxit = 1000,
        tol = 1e-8,
        tol_f64 = 1e-6
      )
    }
  })[["elapsed"]]

  list(fit = fit, elapsed = elapsed)
}

score_fit <- function(dat, fit, ref) {
  pred <- predict(fit, dat$x, type = "response")
  ref_pred <- predict(ref, dat$x, s = fit$lambda, type = "response")
  
  if (dat$family == "gaussian") {
    loss <- mean((pred[, ncol(pred)] - dat$y)^2)
  } else {
    prob <- pmin(pmax(pred[, ncol(pred)], 1e-8), 1 - 1e-8)
    loss <- mean(-2 * (dat$y * log(prob) + (1 - dat$y) * log(1 - prob)))
  }
  
  coef_diff <- mean(abs(as.matrix(coef(ref)) - coef(fit)))
  
  list(
    float64 = isTRUE(fit$float64),
    pred_loss = loss,
    coef_diff = coef_diff,
    max_pred_delta = max(abs(pred - ref_pred)),
    beta_dtype = Rmlx::mlx_dtype(fit$beta)
  )
}

scenarios <- c("dense_gaussian", "gram_gaussian", "binomial")
methods <- c("old_f32", "refined", "full_f64")
rows <- list()

for (scenario in scenarios) {
  dat <- scenario_data(scenario)
  family <- dat$family
  ref0 <- glmnet::glmnet(
    dat$x,
    dat$y,
    family = family,
    alpha = 1,
    nlambda = 12,
    standardize = TRUE,
    thresh = 1e-12,
    maxit = 100000L
  )
  lambda <- ref0$lambda
  ref <- glmnet::glmnet(
    dat$x,
    dat$y,
    family = family,
    alpha = 1,
    lambda = lambda,
    standardize = TRUE,
    thresh = 1e-12,
    maxit = 100000L
  )

  for (method in methods) {
    message("Running ", scenario, " / ", method)
    out <- fit_once(dat, method, lambda)
    score <- score_fit(dat, out$fit, ref)
    rows[[length(rows) + 1L]] <- data.frame(
      scenario = scenario,
      method = method,
      elapsed = out$elapsed,
      float64 = score$float64,
      pred_loss = score$pred_loss,
      coef_diff = score$coef_diff,
      max_pred_delta = score$max_pred_delta,
      beta_dtype = score$beta_dtype
    )
  }
}

results <- do.call(rbind, rows)
