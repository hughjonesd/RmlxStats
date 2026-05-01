fuzz_tier <- skip_fuzz_tests("mlxs_prcomp")

run_prcomp_mc_rep <- function(
  seed,
  scenario,
  n,
  p,
  rank_true,
  rank_fit,
  noise_sd,
  center,
  scale
) {
  x <- prcomp_fuzz_case(
    seed = seed,
    scenario = scenario,
    n = n,
    p = p,
    rank_true = rank_true,
    noise_sd = noise_sd
  )
  fit <- mlxs_prcomp(
    x,
    center = center,
    scale. = scale,
    rank. = rank_fit,
    oversample = 10L,
    n_iter = 2L,
    seed = seed + 1000L
  )
  ref <- stats::prcomp(
    x,
    center = center,
    scale. = scale,
    rank. = rank_fit
  )
  summarise_prcomp_fit(
    x = x,
    fit = fit,
    ref = ref,
    scenario = scenario,
    case_type = "monte_carlo_rep",
    rank_true = rank_true,
    noise_sd = noise_sd,
    center = center,
    scale. = scale
  )
}

summarise_prcomp_mc <- function(results, reps) {
  groups <- split(
    results,
    interaction(results$scenario, results$method, drop = TRUE)
  )
  rows <- lapply(groups, function(group) {
    data.frame(
      case_type = "monte_carlo",
      scenario = group$scenario[[1]],
      n = group$n[[1]],
      p = group$p[[1]],
      nreps = reps,
      rank = group$rank[[1]],
      rank_true = group$rank_true[[1]],
      method = group$method[[1]],
      noise_sd = group$noise_sd[[1]],
      max_sdev_error = max(group$max_sdev_error),
      relative_sdev_rmse = mean(group$relative_sdev_rmse),
      subspace_error = max(group$subspace_error),
      reconstruction_error = mean(group$reconstruction_error),
      reference_reconstruction_error =
        mean(group$reference_reconstruction_error),
      excess_reconstruction_error = max(group$excess_reconstruction_error),
      orthogonality_error = max(group$orthogonality_error),
      explained_variance_error = max(group$explained_variance_error),
      all_finite = all(group$all_finite),
      row.names = NULL
    )
  })
  do.call(rbind, rows)
}

test_that("mlxs_prcomp Monte Carlo fuzz summaries are within tolerance", {
  reps <- if (identical(fuzz_tier, "full")) 60L else 12L
  size_multiplier <- if (identical(fuzz_tier, "full")) 2L else 1L
  specs <- data.frame(
    scenario = c(
      "low_rank", "spiked", "clustered_singular", "sparse_dense"
    ),
    seed0 = c(5001L, 5002L, 5003L, 5004L),
    n = size_multiplier * c(220L, 180L, 160L, 160L),
    p = size_multiplier * c(60L, 300L, 80L, 240L),
    rank_true = c(8L, 8L, 8L, 10L),
    rank_fit = c(8L, 8L, 8L, 10L),
    noise_sd = c(1e-4, 1e-4, 1e-3, 0.01),
    scale = c(FALSE, FALSE, FALSE, TRUE)
  )

  summaries <- vector("list", nrow(specs))
  for (spec_idx in seq_len(nrow(specs))) {
    spec <- specs[spec_idx, ]
    results <- run_mc_reps(
      reps = reps,
      seed0 = spec$seed0,
      rep_fun = run_prcomp_mc_rep,
      label = "run_prcomp_mc",
      reproduce_args = list(
        scenario = spec$scenario,
        n = spec$n,
        p = spec$p,
        rank_true = spec$rank_true,
        rank_fit = spec$rank_fit,
        noise_sd = spec$noise_sd,
        center = TRUE,
        scale = spec$scale
      ),
      scenario = spec$scenario,
      n = spec$n,
      p = spec$p,
      rank_true = spec$rank_true,
      rank_fit = spec$rank_fit,
      noise_sd = spec$noise_sd,
      center = TRUE,
      scale = spec$scale
    )
    summaries[[spec_idx]] <- summarise_prcomp_mc(
      do.call(rbind, results),
      reps = reps
    )
  }
  summaries_df <- do.call(rbind, summaries)

  print(summaries_df, digits = 4)
  write_fuzz_summaries(
    summaries_df,
    suite = "mlxs-prcomp-monte-carlo",
    tier = fuzz_tier
  )

  expect_true(all(summaries_df$all_finite))
  expect_true(all(summaries_df$orthogonality_error <= 5e-6))
  strict <- summaries_df$scenario != "sparse_dense"
  expect_true(all(summaries_df$max_sdev_error[strict] <= 1e-5))
  expect_true(all(summaries_df$subspace_error[strict] <= 1e-5))
  expect_true(all(abs(summaries_df$excess_reconstruction_error[strict]) <=
    1e-6))
})
