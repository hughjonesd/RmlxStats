fuzz_tier <- skip_fuzz_tests("mlxs_prcomp")

test_that("mlxs_prcomp deterministic fuzz cases match stats::prcomp", {
  specs <- data.frame(
    scenario = c(
      "gaussian_full_exact", "low_rank_tall_exact", "low_rank_wide_exact",
      "large_mean_exact", "near_duplicate_exact", "spiked_wide_randomized",
      "spiked_tall_randomized", "sparse_dense_randomized"
    ),
    generator = c(
      "gaussian_full", "low_rank", "low_rank", "large_mean",
      "near_duplicate", "spiked", "spiked", "sparse_dense"
    ),
    n = c(120L, 120L, 40L, 120L, 160L, 100L, 500L, 180L),
    p = c(20L, 20L, 140L, 15L, 30L, 500L, 20L, 220L),
    rank_true = c(20L, 5L, 8L, 5L, 8L, 8L, 8L, 10L),
    rank_fit = c(NA_integer_, NA_integer_, NA_integer_, NA_integer_,
                 NA_integer_, 8L, 8L, 10L),
    noise_sd = c(0, 0, 1e-5, 1e-4, 1e-5, 1e-5, 0.01, 0.01),
    center = c(TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE),
    scale = c(FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE),
    seed = c(3000L, 3001L, 3002L, 3003L, 3004L, 3005L, 3006L, 3007L)
  )

  summaries <- vector("list", nrow(specs))
  for (spec_idx in seq_len(nrow(specs))) {
    spec <- specs[spec_idx, ]
    x <- prcomp_fuzz_case(
      seed = spec$seed,
      scenario = spec$generator,
      n = spec$n,
      p = spec$p,
      rank_true = spec$rank_true,
      noise_sd = spec$noise_sd
    )
    rank_fit <- if (is.na(spec$rank_fit)) NULL else spec$rank_fit
    fit <- mlxs_prcomp(
      x,
      center = spec$center,
      scale. = spec$scale,
      rank. = rank_fit,
      oversample = 10L,
      n_iter = 2L,
      seed = 101
    )
    ref <- stats::prcomp(
      x,
      center = spec$center,
      scale. = spec$scale,
      rank. = if (is.null(rank_fit)) min(spec$n, spec$p) else rank_fit
    )
    summaries[[spec_idx]] <- summarise_prcomp_fit(
      x = x,
      fit = fit,
      ref = ref,
      scenario = spec$scenario,
      case_type = "deterministic",
      rank_true = spec$rank_true,
      noise_sd = spec$noise_sd,
      center = spec$center,
      scale. = spec$scale
    )
  }
  summaries_df <- do.call(rbind, summaries)
  write_fuzz_summaries(
    summaries_df,
    suite = "mlxs-prcomp-deterministic",
    tier = fuzz_tier
  )

  sdev_error <- summaries_df[
    summaries_df$target == "pca_sdev" & summaries_df$measure == "error",
  ]
  subspace <- summaries_df[summaries_df$target == "subspace", ]
  excess_recon <- summaries_df[
    summaries_df$target == "reconstruction" &
      summaries_df$measure == "delta",
  ]
  orthogonality <- summaries_df[summaries_df$target == "orthogonality", ]
  finite <- summaries_df[summaries_df$target == "finite", ]
  exact <- sdev_error$method == "exact"
  randomized <- sdev_error$method == "randomized"
  # Large means, duplicate columns, wide low-rank fits, and sparse-ish
  # randomized fits are kept as tracked stress diagnostics. The strict gates
  # below apply to well-identified reference components.
  strict_exact <- sdev_error$scenario == "gaussian_full_exact"
  strict_randomized <- randomized &
    sdev_error$scenario != "sparse_dense_randomized"
  expect_true(all(as.logical(finite$value)))
  expect_true(all(orthogonality$value <= 5e-6))
  expect_true(all(sdev_error$value[exact & strict_exact] <= 1e-5))
  strict_exact <- subspace$scenario == "gaussian_full_exact"
  expect_true(all(subspace$value[subspace$method == "exact" & strict_exact] <=
    1e-5))
  expect_true(all(abs(excess_recon$value[excess_recon$method == "exact"]) <=
    1e-5))
  expect_true(all(sdev_error$value[strict_randomized] <= 1e-5))
  strict_randomized <- subspace$method == "randomized" &
    subspace$scenario != "sparse_dense_randomized"
  expect_true(all(subspace$value[strict_randomized] <= 1e-5))
})

test_that("mlxs_prcomp deterministic metamorphic properties hold", {
  x <- prcomp_fuzz_case(
    seed = 4001L,
    scenario = "spiked",
    n = 160L,
    p = 45L,
    rank_true = 8L,
    noise_sd = 1e-4
  )
  rank_fit <- 6L
  fit <- mlxs_prcomp(
    x,
    center = TRUE,
    scale. = FALSE,
    rank. = rank_fit,
    oversample = 10L,
    n_iter = 2L,
    seed = 202
  )

  set.seed(4002L)
  row_perm <- sample(seq_len(nrow(x)))
  row_fit <- mlxs_prcomp(
    x[row_perm, , drop = FALSE],
    center = TRUE,
    scale. = FALSE,
    rank. = rank_fit,
    oversample = 10L,
    n_iter = 2L,
    seed = 202
  )
  expect_equal(as.numeric(row_fit$sdev), as.numeric(fit$sdev),
               tolerance = 1e-6)
  expect_lte(
    prcomp_projector_error(row_fit$rotation, fit$rotation),
    2e-6
  )

  col_perm <- sample(seq_len(ncol(x)))
  col_fit <- mlxs_prcomp(
    x[, col_perm, drop = FALSE],
    center = TRUE,
    scale. = FALSE,
    rank. = rank_fit,
    oversample = 10L,
    n_iter = 2L,
    seed = 202
  )
  col_rotation <- matrix(NA_real_, nrow = ncol(x), ncol = rank_fit)
  col_rotation[col_perm, ] <- as.matrix(col_fit$rotation)
  expect_lte(
    prcomp_projector_error(col_rotation, fit$rotation),
    2e-6
  )

  same_seed_fit <- mlxs_prcomp(
    x,
    center = TRUE,
    scale. = FALSE,
    rank. = rank_fit,
    oversample = 10L,
    n_iter = 2L,
    seed = 202
  )
  reproducibility_error <- max(
    max(abs(as.numeric(same_seed_fit$sdev) - as.numeric(fit$sdev))),
    prcomp_projector_error(same_seed_fit$rotation, fit$rotation)
  )
  expect_lte(reproducibility_error, 1e-7)

  rank_ladder <- c(2L, 4L, 6L)
  recon <- vapply(rank_ladder, function(rank_value) {
    ladder_fit <- mlxs_prcomp(
      x,
      center = TRUE,
      scale. = FALSE,
      rank. = rank_value,
      oversample = 10L,
      n_iter = 2L,
      seed = 202
    )
    prcomp_reconstruction_error(
      x,
      as.matrix(ladder_fit$rotation),
      center = TRUE,
      scale = FALSE
    )
  }, numeric(1))
  monotonicity_error <- max(diff(recon), 0)
  expect_lte(monotonicity_error, 1e-8)

  write_fuzz_summaries(
    fuzz_metric_rows(
      list(
        case_type = "metamorphic",
        scenario = "row_column_rank_seed",
        n = nrow(x),
        p = ncol(x),
        rank = rank_fit,
        rank_true = 8L,
        method = fit$method,
        noise_sd = 1e-4
      ),
      measure     = c("error",     "diagnostic",  "diagnostic",     "diagnostic"),
      target      = c("subspace",  "monotonicity", "reproducibility", "finite"),
      source      = c("mlx",       "mlx",          "mlx",             "mlx"),
      baseline    = c("reference", "ideal",        "same_seed",       NA),
      aggregation = c("max",       "max",          "value",           "all"),
      value = c(
        max(
          prcomp_projector_error(row_fit$rotation, fit$rotation),
          prcomp_projector_error(col_rotation, fit$rotation)
        ),
        monotonicity_error,
        reproducibility_error,
        as.numeric(all(is.finite(c(recon, reproducibility_error))))
      )
    ),
    suite = "mlxs-prcomp-deterministic",
    tier = fuzz_tier
  )
})
