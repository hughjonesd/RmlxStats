# Linear-model precision and repeated timings

Measured on an Apple M2 MacBook Air (16 GB), R 4.6.1, macOS 27.0.1.
Each fixture/method has 30 timed repetitions after an untimed warm-up.
Methods are randomly ordered within each round. GPU jobs were run alone.
Input construction and transfers are outside the timer; coefficients,
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

## Warning diagnostic

`mlxs_lm_precision(fit)` makes an optional CPU float64 pass over the original
model frame and evaluates `||R^-T X' residual|| / sigma`. It warns above
one joint standard-error unit by default; `tol` adjusts the threshold.
Original factor contrasts and square-root weights are preserved.
It performs no second QR factorization and adds no work to ordinary fits.
The score is approximate: an inaccurate QR factor can mislead it, and it
cannot recover input precision lost before the data reached the fit.

### Diagnostic cost versus a complete formula fit

`dev/benchmark-lm-precision-cost.R` compares the diagnostic with an actual
`mlxs_lm(y ~ ., data)` call, including model-frame construction, the design
matrix and array conversion. Fit outputs are evaluated inside the timer;
the diagnostic's scalar result forces its evaluation. Both use the same
data, with 30 warmed, randomly ordered repetitions per method.

| Rows | Columns including intercept | Fit median | Diagnostic median | Added time |
|---:|---:|---:|---:|---:|
| 1,000 | 6 | about 3 ms | about 1 ms | about 33% |
| 200,000 | 51 | 72 ms | 13 ms | 18% |
| 1,000,000 | 51 | 351.5 ms | 60 ms | 17% |
| 60,000 | 201 | 66 ms | 14 ms | 21% |

The small fit uses CPU QR; the others use Cholesky QR. Millisecond timer
resolution limits the small-case precision. Raw timings and percentile
summaries are in `diagnostic-cost/`. The formula fit times are greater than
the earlier solver timings because input preparation is included here.
These are separate fit/check measurements; percentages divide their medians.

The diagnostic computes two matrix-vector products, reductions and one
triangular solve: O(np + p²) arithmetic, plus O(np) design reconstruction
and copying. Cholesky QR requires matrix-matrix products and small dense
factorizations: O(np² + p³), plus its own O(np) input preparation. The
diagnostic therefore saves a factor of approximately p in the dominant
arithmetic as n grows relative to p. Actual time does not follow this ratio:
the fit uses float32 GPU operations; the check uses float64 CPU operations;
both incur allocation and memory traffic. At these sizes, data preparation
accounts for much of the full formula fit's time.

The check is reasonably cheap relative to the complete formula API, but
is not free. Against the earlier array-only solver timings, its cost is a
much larger fraction. It remains optional so applications can choose when
to pay for the additional assurance.
