# Linear-model precision and repeated timings

Measured on an Apple M2 MacBook Air (16 GB), R 4.6.1, macOS 27.0.1.
Each fixture/method has 30 timed repetitions after an untimed warm-up.
Methods are randomly ordered within each round. GPU jobs were run alone.
For these array-solver measurements, input construction and transfers
are outside the timer; coefficients,
fitted values and residuals are evaluated inside it.

The baseline is RmlxStats `547a1a6`; the candidate adds the missing residual
correction. Both use the merged Rmlx GPU QR R implementation at `3f1d90e`,
loaded against the installed, working Rmlx runtime. This isolates the R-level
change rather than benchmarking an unrelated rebuild of the MLX runtime.
The fixtures have 51 columns including an intercept. CPU64 uses the original
double input data, rather than promoting already-rounded float32 arrays.

| Fixture | GPU before | GPU after | CPU32 after | CPU64 after |
|---|---:|---:|---:|---:|
| IID, 200,000 rows | 19.5 ms | 22 ms | 108.5 ms | 194 ms |
| Low-noise correlated, 200,000 rows | 21.5 ms | 25 ms | 110.5 ms | 202.5 ms |
| IID, 1,000,000 rows | 82 ms | 87 ms | 657 ms | 1,078.5 ms |

These are medians. The before/after GPU 10th–90th percentile ranges overlap;
the small timing differences should not be treated as precise overhead
estimates. The raw timings and percentile summaries are adjacent CSV files.
The corrected million-row fit retains a 7.6-fold advantage over CPU32.

In the difficult fixture, two predictor columns are correlated, the design
condition number is approximately 100, and noise SD is 0.0001. The maximum
coefficient error, divided by the corresponding CPU64 reference standard
error, falls from **128.68 to 0.216**. CPU32 gives 4.647; CPU64 gives
0.000000078. Absolute coefficient error falls from 0.00002888 to
0.00000006981. Ordinary IID accuracy is unchanged. These are deterministic
precision comparisons for one seed, not 30 independent numerical fixtures.

The earlier experimental `beta + delta` update gave 0.084 standard errors.
The implementation in this PR instead updates the corrected right-hand
side and solves again, preserving the existing Rmlx QR metadata convention.
Different float32 rounding gives the reported 0.216 standard errors.

Full-tier fuzz tests additionally exercise explicit TSQR and weighted GPU
fits. Their worst coefficient errors are 0.317 and 0.169 standard errors;
all three residual sums of squares differ from their references by less
than 0.001%. The tests emit their metrics through the existing fuzz output
infrastructure. Before the correction, Cholesky and TSQR both fail the
coefficient and residual precision assertions.

## Reproduce

Run `Rscript dev/benchmark-lm-precision.R` from the repository root with a
working merged Rmlx installation. It defaults to 30 rounds and writes to
`dev/lm-precision-benchmark/run`. Optional environment variables:

- `RMLX_AUDIT_SOURCE`: Rmlx source tree to load without recompiling.
- `RMLX_STATS_SOURCE`: RmlxStats checkout to load (default `.`).
- `RMLX_PRECISION_ITERATIONS`: number of timing rounds (default `30`).
- `RMLX_PRECISION_OUTPUT`: output directory.
- `RMLX_PRECISION_METHODS`: comma-separated subset of
  `current,cpu32,float64`.

Run the integrated tests with `RMLXSTATS_RUN_FUZZ=fast` or `full` and
`devtools::test(filter = "fuzz-lm-det")`. Set `RMLXSTATS_FUZZ_OUT` to save
their results. The fast tier adds two low-noise cases; full adds one weighted
case. Benchmarks are manual and do not impose speed assertions on CI.
Run `Rscript dev/lm-precision-benchmark/plot.R` to recreate `tradeoff.png`.

## Automatic precision checks

`mlxs_lm()` and `mlxs_glm()` now check the final fit automatically. The
`precision_tol` thresholds combined numerical coefficient error relative to
statistical uncertainty; both default to `0.1`. For one coefficient the score
is estimated error divided by its standard error. For independent estimates,
it is `sqrt(sum((error / SE)^2))`; correlated estimates use their covariance
to avoid counting overlapping directions twice. `NULL` disables the check
and its CPU data pass.
`NULL` is the usual R convention for an absent optional operation.
Checked fits store the scalar score as `precision_error`, allowing later
comparison with a stricter threshold without another check. Disabled fits
have no score. Fits without residual degrees of freedom or residual scale
store `NA`.

The LM check evaluates `||R^-T X' W residual|| / sigma` in CPU float64, using
the original data, prior weights and existing weighted QR factor. The GLM
check evaluates `X' W (y-mu) mu_eta / variance(mu)` at the final coefficients,
using the family, original data and prior weights; its transformed norm is
scaled by dispersion. Gaussian dispersion is recomputed from float64
residuals. Other GLMs use their fitted dispersion, with one for binomial and
Poisson models.

Both checks reuse the original design matrix already constructed by the
formula interface, perform no second QR, and run once after fitting. The
last GLM QR factor uses the preceding iteration's working weights, so its
score is a local approximation to the size of a further scoring step. It
can detect numerical error or incomplete convergence; it cannot distinguish
them or exclude separation. An inaccurate QR factor can mislead either
check, and precision lost before supplying the data cannot be recovered.
A small score is not an accuracy certificate.

The 0.1 threshold seeks numerical error substantially below sampling
uncertainty. It deliberately warns on the corrected low-noise fixtures:
although their worst coefficient errors are below 0.5 reference SE, their
joint scores remain around 0.65–0.74. Users can relax the threshold to one
unit, or disable it for workloads such as repeated bootstrap fits.

No diagnostic function is exported. The internal checker is shared by the
two formula interfaces; the array solver used inside IRLS remains unchanged
apart from the residual correction described above.

### Cost versus complete formula fits

`dev/benchmark-lm-precision-cost.R` times actual formula fits with
`precision_tol = NULL` versus the default `0.1`. It includes model-frame
construction, the design matrix and array conversion, and evaluates fit
outputs inside the timer. There are 30 warmed, randomly ordered repetitions
per method. Raw timings and percentiles are in `automatic-check-cost/`.
These costs include the check inside the fit, rather than adding together
separate fit and diagnostic timing medians.

| Model and dimensions | Check disabled | Default check | Added time |
|---|---:|---:|---:|
| LM, 200,000 × 51 | 67 ms | 76.5 ms | 14.2% |
| LM, 1,000,000 × 51 | 347.5 ms | 395 ms | 13.7% |
| LM, 60,000 × 201 | 64.5 ms | 73.5 ms | 14.0% |
| Gaussian GLM, 200,000 × 51 | 101 ms | 112 ms | 10.9% |
| Binomial GLM, 200,000 × 51 | 563 ms | 571 ms | 1.4% |
| Poisson GLM, 200,000 × 51 | 192.5 ms | 201 ms | 4.4% |

The 1,000 × 6 CPU fit takes approximately 3 ms unchecked and 4 ms checked;
millisecond timer resolution limits that small-case comparison. Overhead
percentages compare medians; raw percentile ranges show the timing variability.

The check needs O(np + p²) arithmetic and O(np) conversion and memory
traffic. Cholesky QR needs O(np² + p³), plus O(np) input preparation. Actual
time also depends on CPU float64 versus GPU float32 throughput, allocation,
and retained arrays. A final check is a smaller fraction of an iterative
GLM fit than of a single linear solve in these measurements.
