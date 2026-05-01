# Fuzz Testing Best Practices for Statistical Libraries

## Executive summary

For a statistical library, ordinary unit tests are necessary but not
sufficient. The most effective quality strategy is a layered one:
deterministic contract tests for API and numerical sanity;
simulation-based tests for statistical operating characteristics such as
bias, interval coverage and Type I error; and oracle-weak techniques
such as metamorphic and differential testing for cases where a single
“correct answer” is unavailable or expensive to compute. This is the
pattern that emerges both from the software-testing literature and from
statistical-software reliability work, including certified reference
datasets, simulation-study guidance, metamorphic-testing research, and
current practice in major libraries.
citeturn17search0turn18search8turn18search2turn21search10turn16search2turn27search1turn19search1

The highest-yield bugs in statistical routines tend to come from a
relatively small set of stressors: ill-conditioning, exact or near rank
deficiency, quasi- or perfect separation, extreme class imbalance, heavy
tails, leverage points, small-sample regimes, missingness, grouped or
dependent observations, sparse high-dimensional inputs, and version- or
backend-sensitive reproducibility problems. Historical reliability
studies and modern issue trackers both show that these are not edge
curiosities; they are the places where real libraries have produced
incorrect answers, unstable warnings, or silent inconsistencies.
citeturn20search3turn20search0turn28search9turn28search1turn34search0turn34search1turn26search4turn26search5turn10search1turn13search7turn35search0

A good fuzzing design therefore should not spend most of its budget on
broad, uniform random IID inputs. It should stratify the input space by
risk, oversample known pathological regions, preserve shrunk
counterexamples as regression fixtures, and allocate runtime by tier:
fast deterministic checks per pull request, broader property and
differential campaigns nightly, and Monte Carlo operating-characteristic
studies weekly or before release. For Monte Carlo claims, the library
should track the uncertainty of the test itself, not only the statistic
under test.
citeturn16search2turn21search10turn18search5turn19search1turn24search0turn10search2turn13search1turn7search3

For the `mlxs_*` family specifically, the priority order should be `glm`
and `cv_glmnet` first, then `lm` and `glmnet`, then `pca`, then
`bootstrap`. That ordering is driven by a combination of oracle
weakness, optimisation complexity, sensitivity to data pathologies, and
the frequency with which bugs arise from folds, convergence,
separability, and non-identifiability.
citeturn3search5turn11search1turn26search8turn10search1turn34search1turn34search7turn36search1turn29search7

## Properties, oracles and scoring

The table below is the practical core of a fuzzing programme for a
statistical library. The main design principle is that each property
should have a **measurement rule**, an **acceptable tolerance**, and a
**failure interpretation**. Statistical tests should almost never use
raw point targets without accounting for Monte Carlo error or
identification issues.

| Property | What to test | Concrete methodology | Useful metrics | Typical failure modes |
|----|----|----|----|----|
| Unbiasedness | Estimator centred on truth under a correctly specified DGP | Simulate under known parameters; compare mean estimate to target over replications | Bias, relative bias, RMSE, empirical SE | Systematic drift, wrong offset handling, scaling bugs |
| Confidence-interval coverage | Nominal intervals contain truth at the advertised rate | Repeated simulation under simple well-specified models; track contain/not-contain | Empirical coverage, average width, under/over-coverage | Incorrect SEs, bad quantiles, Hessian inversion problems |
| Type I error | Null rejected at the nominal level under the null | Generate null datasets; run inferential routine many times | Empirical size at α, Monte Carlo CI for size | Inflated false positives, flawed df accounting, anti-conservative covariance |
| Type II error and power | Rejection probability increases under alternatives | Generate alternatives on an effect-size grid | Power curves, minimum detectable effect, monotonicity in n | Low sensitivity, solver failures on signal-rich cases |
| Calibration | Probabilities or intervals correspond to observed frequencies | Reliability diagrams, PIT/rank checks, simulation-based calibration where applicable | ECE, Brier score, PIT uniformity, SBC rank histograms | Overconfident predictions, biased posterior/probability outputs |
| Consistency | Accuracy improves as sample size grows | Nested sample-size ladder with fixed DGP | Slope of log-error vs log-n, monotone error reduction | Incorrect asymptotics, scaling breakdowns |
| Numerical stability | Output robust to algebraically equivalent formulations | Compare centred/scaled, permuted, weighted-equivalent and high-precision/reference variants | Relative error, NaN/Inf rate, objective agreement, projector distance | Catastrophic cancellation, conditioning sensitivity, unstable branching |
| Convergence | Optimiser terminates for the right reasons | Track status, gradient/KKT conditions, path monotonicity where applicable | Iteration count, final gradient norm, duality gap, deviance/objective change | False convergence, oscillation, non-monotone deviance, infinite loops |
| Reproducibility | Same seed and same folds imply same result | Repeat exact runs under controlled random state and thread settings | Bitwise equality or tolerance-based equality, split stability | Hidden randomness, backend instability, non-stable sorting |
| API robustness | Inputs are validated consistently and outputs honour the contract | Property-based generation of shapes, dtypes, names, sparse/dense, weights, offsets | Exception category/message, shape invariants, metadata propagation | Silent coercion, wrong shape, inconsistent warnings |
| Edge-case handling | Degenerate but valid or invalid inputs behave predictably | Hand-crafted corpus plus fuzz around boundaries | No crash, correct warning/error, documented return form | Segfaults, nonsense estimates, contradictory diagnostics |

This framing is consistent with simulation-study guidance in the
statistical literature, Monte Carlo error guidance, metamorphic-testing
work, calibration literature, and library documentation for common
statistical routines. In practice, three rules matter most. First,
report a Monte Carlo uncertainty band for every operating-characteristic
result. Secondly, do not compare non-identifiable quantities directly:
for rank-deficient regressions and PCA sign ambiguity, compare fitted
values, subspaces, objectives, or prediction outputs rather than raw
coefficients or loadings. Thirdly, promote every real failure into a
small deterministic regression fixture.
citeturn16search2turn21search10turn22search0turn22search7turn18search5turn18search2turn38search1turn26search5turn36search1

For simulation-based properties, the test harness should treat the test
result itself as a binomial estimate. If the target is a rate (p)
estimated from (R) replications, the Monte Carlo standard error is
approximately (). As a rough planning rule, estimating a rate to within
±0.01 at 95% confidence needs about 1,825 replications when (p ), and
about 9,604 replications in the conservative (p=0.5) worst case. That is
why PR-time checks should use coarse screens and why tight coverage
claims belong in nightly or release-tier jobs, not in every commit gate.
citeturn21search10turn33calculator2turn33calculator0

A robust oracle hierarchy is usually: certified reference data first;
closed-form or high-precision results second; differential comparison
against mature implementations third; metamorphic relations fourth; pure
crash-freeness last. The entity\[“organization”,“National Institute of
Standards and Technology”,“us standards agency”\] Statistical Reference
Datasets are especially valuable because they provide certified values
and explicit difficulty gradings for regression and related statistical
tasks, including well-known hard cases such as Longley and Wampler-type
datasets. citeturn27search1turn27search0turn27search3turn20search2

## Data generation and coverage strategy

Uniform random sampling is almost always the wrong default for
statistical fuzzing. Most production failures live in small, structured
pockets of the input space. The fuzzer should therefore generate from
*families* of data-generating processes with explicit control
parameters, and then stratify or importance-sample over those control
parameters. That approach follows both the ADEMP-style simulation
literature and the metamorphic and differential-testing literature:
define aims, vary the factors that matter, and allocate more budget to
regions where discrepancies or pathological conditioning are likely.
citeturn16search2turn18search5turn19search1turn21search10

| Generator family | Parameterisation | Why it matters | Expected failure modes |
|----|----|----|----|
| IID Gaussian baseline | (X\_{ij}N(0,1)), homoskedastic noise | Sanity reference for bias, coverage, power | Basic implementation errors |
| Heavy tails | Student-t with ({1,2,3,5,10}); contaminated normal | Stress SEs, outlier sensitivity, bootstrap instability | Exploding variance, unstable CI coverage |
| Skewness | Log-normal, Gamma, shifted exponentials | Tests symmetry assumptions and scale transforms | Biased intervals, poor Wald approximations |
| Heteroscedasticity | (\_i²⁼2(1+c | x_i | )^2) or (\_i²⁼2 e^{x_i}) |
| Correlated predictors | AR(1) ( {0.5,0.9,0.99}), block equicorrelation, factor models | Reveals instability and selection path dependence | Multicollinearity, inflated SEs, path instability |
| Near-collinearity | Singular values decaying geometrically to (10^{-12}) | Targets numerical precision loss | Aliasing, nonsense coefficients, convergence warnings |
| Exact rank deficiency | Duplicate columns, linear combinations, intercept + full dummy set | Non-identifiability | Wrong rank, invalid df, silent nonsense output |
| Separability | Logistic signal scaled until classes are (quasi-)separable | Canonical GLM failure zone | Infinite coefficients, false convergence |
| Outliers and leverage | Single or few rows with extreme (x) and/or (y) | Classical regression stressor | Dominated fits, breakdown of covariance estimates |
| Missingness | MCAR and MAR masks at 1%, 5%, 20%, 50% | API and statistical handling of NA | Silent dropping, shape bugs, incorrect warnings |
| Time dependence | AR(1), ARMA-like residuals, blocked dependence | Exposes misuse of IID assumptions | Invalid CV, anti-conservative inference |
| Clustered data | Random-intercept or multi-level grouping | Tests grouped sampling and leakage control | Mis-sized SEs, fold leakage |
| Adversarial structure | Constant columns, zero weights, all-one/all-zero outcomes, single-class folds, sparse matrices, reordered metadata | Targets branch logic and validation | Crashes, wrong exceptions, silent coercion |

The most useful practical design is a **risk stratification grid** over
a small number of axes: sample-size regime (n/p), condition number, tail
weight, class imbalance, missingness rate, dependence structure,
density/sparsity, and optimisation difficulty. A sensible starting
allocation is 40% of budget to boundary regions, 40% to medium-risk
structured cases, and only 20% to broad random exploration. For
optimisation-heavy routines, importance-sample especially near ((X)),
class prevalence below 1% or above 99%, sparse densities below 1%, and
logits or means close to floating-point underflow/overflow boundaries.
citeturn20search0turn27search0turn19search1turn18search2

Metamorphic testing is particularly powerful for statistical routines
because exact oracles are frequently unavailable. High-value metamorphic
relations include row-permutation invariance; column-permutation
invariance with inverse permutation of outputs; invariance of fitted
values when equivalent weighted and duplicated datasets are used;
preservation of regression predictions under reparameterisations that
leave the linear predictor unchanged; monotone reduction in PCA
reconstruction error as the retained rank increases; and invariance of
deterministic CV results under row permutation when the fold assignment
is permuted with the rows. For PCA, compare subspaces or reconstruction
loss rather than raw sign-sensitive loadings.
citeturn18search8turn18search2turn18search6turn18search9turn36search1

Differential testing should compare the system under test against
multiple references, not only one. For small well-conditioned cases,
compare against [`stats::lm`](https://rdrr.io/r/stats/lm.html)/`glm` in
R, `statsmodels`, or exact SVD/QR computations. For penalised paths,
compare predictions, objective values and active sets on carefully
chosen canonical cases, not necessarily internal coefficient-path
details. For numerical kernels and scalar special-function calls,
compare against high-precision libraries or certified references. The
key lesson from the differential-testing literature is that
disagreements across supposedly synonymous computations are often
genuine bug leads rather than harmless noise.
citeturn19search1turn27search1turn38search0turn37search4turn8search6turn36search0

## Literature and ecosystem evidence

The research literature points in a clear direction. Property-based
testing, beginning with QuickCheck, established the practical value of
describing invariants and then searching for counterexamples with rich
generators. Metamorphic testing later became the standard answer to
weak-oracle domains such as statistical and machine-learning software.
For statistical routines specifically, recent work on multiple linear
regression showed that metamorphic relations can detect faults that
ordinary example-based testing misses. Differential testing of numerical
libraries has likewise shown that disagreement-based fuzzing can uncover
large numbers of real defects across mature libraries.
citeturn17search0turn17search2turn18search8turn18search1turn18search2turn19search1

A parallel literature in statistics emphasises how to validate *methods*
rather than only code. Simulation-study guidance recommends explicit
aims, explicit data-generating mechanisms, explicit estimands and
explicit performance measures, plus reporting of Monte Carlo error.
Calibration work, especially simulation-based calibration, contributes a
reusable idea for any inference engine that produces distributions or
probabilities: validation should check whether nominal uncertainty is
empirically correct over repeated draws from the assumed generative
process. Reliability studies of statistical software, from the 1980s
through the 2010s, repeatedly found that benchmark datasets and hard
numerical cases expose real failures even in widely used packages.
citeturn16search2turn21search10turn22search0turn22search7turn20search3turn20search0turn28search1turn28search6turn28search9

The strongest external oracle available to many classical routines
remains the NIST reference-data ecosystem. The certified datasets cover
univariate summaries, ANOVA, linear regression, nonlinear regression,
and MCMC-oriented checks; they include graded difficulty and are
explicitly designed to stress algorithms that otherwise appear correct
on easy integer or textbook examples. For a statistical library, using
these datasets is not optional good citizenship; it is a direct route to
catching silent numerical failures.
citeturn27search1turn27search3turn27search0turn20search2turn27search5

The practices of major libraries broadly line up with this literature,
although with different emphases. The table below summarises the
highest-confidence patterns visible in official repositories,
documentation and issue trackers.

| Library | Observable current practice | Concrete examples worth copying | Examples of relevant failures or bug signals |
|----|----|----|----|
| R base/stats | Source tree with `tests/` and regression tests; `R CMD check`; examples and saved-output comparisons; explicit `singular.ok` and rank/alias handling in `lm`; `glm` exposed through `glm.fit` | Treat `tests/` as a long-lived regression corpus; use examples as executable specs; preserve aliased/rank information | Broad bug-tracker infrastructure exists; hard failures tend to surface in regression tests and numerical edge cases |
| MASS | Classical package with `tests/` in source tree | Keep stable hand-written statistical test corpus for mature routines | Public mirror is read-only, so issue evidence is less visible than in GitHub-native projects |
| lme4 | Mix of classic tests and `testthat`; GitHub Actions for R-CMD-check, all-OS checks, R-hub and coverage | Combine traditional statistical regression tests with modern CI and coverage | Open issues on convergence disagreements, singular fits, PSD Cholesky fallback and `glmer.nb()` edge cases |
| glmnet | Source + vignettes + official changelog/NEWS; algorithm docs explain path-wise coordinate descent and convergence | Keep precise changelog of bug-inducing edge cases and use them as future fixtures | Official changelog records bugs involving constant `y`, `lambda.interp` NaNs, exact prediction, sparse-X Cox paths, single-row prediction, infinite loops and survival-function fixes |
| caret | GitHub Actions for R-CMD-check and coverage; dedicated `RegressionTests` directory; package in maintenance mode | Separate long-lived regression fixtures from ordinary tests | Issues include `groupKFold` returning the wrong number of folds and inconsistent NA handling |
| scikit-learn | Large `pytest` suite, common estimator checks, seed control, multiple CI directories/workflows | Use reusable contract tests across estimators and explicit seed control | Issues around PCA interpretation, `GroupKFold` reproducibility, `groups` propagation in nested CV and other CV semantics |
| statsmodels | `pytest` conventions documented; extensive module-specific tests; multiple workflows including wheels, docs, CodeQL and Pyodide; explicit warning/exception taxonomy | Treat warnings such as perfect separation and collinearity as part of the contract | Issues include perfect collinearity, rank-deficient Wald tests, wrong-ordered FDR outputs above 1, convergence warnings and version-compatibility failures |

These patterns are directly visible in the official documentation and
repositories hosted on entity\[“company”,“GitHub”,“software hosting
company”\]. The entity\[“organization”,“R Core Team”,“r language
maintainers”\] documentation exposes `lm` rank and alias handling and
`glm`’s fitting interface; `prcomp` is explicitly SVD-based.
`scikit-learn` documents reusable estimator checks and explicit test
seeding; `statsmodels` documents `pytest` conventions and exposes
dedicated convergence, collinearity and perfect-separation warnings;
`lme4` documents its mixed classic/`testthat` strategy and its current
CI workflows; `caret` exposes both CI and a dedicated regression-test
corpus; and `glmnet` exposes both the algorithm and a detailed bug-fix
history in its changelog.
citeturn38search0turn38search1turn37search4turn37search0turn24search0turn6search1turn2search13turn8search6turn10search2turn10search3turn4search1turn13search1turn35search0turn11search1

A practical implication follows. The best libraries do **not** rely on
“more random tests” alone. They combine reusable contract tests,
hand-curated regression fixtures from past bugs, clearly documented
warnings for non-regular cases, and CI tiers that acknowledge runtime
constraints. Your own library should do the same.
citeturn24search0turn10search2turn13search1turn7search3

## Recommended fuzz campaigns for `mlxs_*`

The most useful design for `mlxs_*` is to combine one **fast
deterministic suite**, one **priority fuzz suite**, and one
**statistical validation suite** for each function. The tables below
give concrete starting points. Unless your semantics differ materially
from the corresponding standard routines, the recommended oracles and
generators should transfer well. Where semantics differ, adapt the
**asserted property**, not the stress families themselves.
citeturn38search0turn37search4turn37search0turn11search1turn29search7

### Function-specific test matrix

| Function | Fast PR-time tests | Nightly fuzz tests | Weekly/release statistical tests | High-value assertions |
|----|----|----|----|----|
| `mlxs_lm` | NIST/Longley/Wampler-style regressions; API validation; row/column permutation invariance | Near-collinearity, exact rank deficiency, leverage points, heteroscedasticity, missingness behaviour | Bias, CI coverage and Type I error under clean Gaussian and heteroscedastic settings | Correct rank/alias handling; fitted values invariant to equivalent parameterisations; no silent nonsense on singular designs |
| `mlxs_glm` | Canonical small Gaussian/binomial/Poisson cases; warning/error contracts | Quasi-separation, rare events, offsets, weights, all-zero/all-one outcomes, sparse design, high leverage | Empirical size and power; interval coverage under regular cases; calibration of probabilities or intervals | Means/probabilities in valid range; convergence status matches pathology; no false confident success on separation |
| `mlxs_pca` | Exact low-rank matrices; full-SVD comparisons on small inputs | Near-duplicate columns, rank deficiency, sparse inputs, randomised solver reproducibility | Eigenvalue recovery under spiked covariance; reconstruction-error monotonicity | Orthonormal components; sign-agnostic subspace agreement; explained variance ordering and sane sums |
| `mlxs_glmnet` | Orthonormal-Gaussian cases with soft-threshold oracle; API/contracts | Block correlation, extreme (), sparse high-(p), constant/near-constant response, custom lambda grids | Prediction stability and objective agreement across regimes; path behaviour on benchmark families | Decreasing lambda grid; finite predictions; correct handling of known historical edge cases |
| `mlxs_cv_glmnet` | Fixed `foldid` reproducibility; fold-shape validation | Row permutation with permuted folds; class imbalance; leakage-prone grouped data via custom fold IDs | Distribution of selected () across seeds and DGPs; out-of-sample risk bias | Same data + same folds + same seed gives same result; `lambda.1se` respects one-SE rule; impossible folds fail clearly |
| `mlxs_bootstrap` | Basic resampling invariants; seed reproducibility; degenerate statistic cases | Percentile/basic/BCa intervals on skew, heavy-tail and regression statistics; stratified/clustered variants if supported | Coverage studies for mean, median and regression coefficients under several DGPs | Resamples of correct size with replacement; collapsed interval for constant statistics; endpoints ordered and finite |

### Suggested parameter ranges and seed set

Use a small fixed seed bank for reproducibility and shrinkage:
`1, 17, 2718, 314159, 8675309`. For PR-time tests, select one or two
seeds per stratum; for nightly tests, rotate through the full bank; for
weekly tests, expand by deterministic derivation from the test case
hash. This mirrors the general reproducibility practice in mature
libraries that expose explicit seed controls.
citeturn6search1turn24search0

For `mlxs_lm`, start with (n {8, 16, 32, 64, 256}), (p {1, 2, 5, 10,
(50,n-1), n}), and add invalid cases such as (p\>n) or exact duplicate
columns if the API is supposed to reject them. Generate (X) from IID
normal, AR(1) with ({0.5,0.9,0.99}), and exact or near-linear
combinations. Noise should include homoskedastic Gaussian and
heteroscedastic variants. Assert: rank equals reference rank; `aliased`
or dropped coefficients are handled consistently; fitted values are
invariant under row permutation; and on well-conditioned small problems
coefficients and residual scale match a QR/SVD reference within tight
tolerance. On singular problems, never require equality of raw
coefficients; require agreement of fitted values and residual sum of
squares instead.
citeturn38search0turn38search1turn27search0turn26search5

For `mlxs_glm`, cover at least Gaussian, binomial and Poisson semantics
if supported. Use (n {20, 50, 200, 1000}), (p {1, 5, 20, 100}),
prevalence targets from 1% to 99% for binomial cases, and
offsets/exposures for Poisson cases. Deliberately generate
quasi-separable and separable logistic problems by increasing the signal
and reducing overlap. Include invalid extremes such as all-zero or
all-one outcomes and check that the function warns or errors rather than
silently claiming convergence. When applicable, monitor the final
gradient, objective or deviance and compare predictions against
reference implementations on regular cases. For inferential checks, use
easy, regular settings only; non-regular cases belong to
warning-contract tests, not coverage targets.
citeturn37search4turn3search5turn8search6turn10search1turn26search7

For `mlxs_pca`, test shapes ((20,5)), ((50,50)), ((100,500)), and
((500,20)) with isotropic, spiked and exact low-rank covariance
structures. Use exact-rank-(k) synthetic matrices where the residual
after projecting to (k) components should be numerically tiny, then add
tiny perturbations to reveal loss of orthogonality and solver
instability. Compare subspaces through projector matrices or absolute
loading correlations because PCA signs are not identifiable. Assert that
components are orthonormal, singular values or explained variances are
ordered, and reconstruction error is non-increasing in the retained
rank. With randomised solvers, fixed seeds should produce reproducible
outputs within tolerance, not necessarily bitwise identity across all
backends. citeturn36search1turn34search0

For `mlxs_glmnet`, use (n {30,100,300}), (p {10,100,1000}), ({0, 0.1,
0.5, 0.9, 1}), and sparse design densities in ({0.01,0.1,0.5}). Include
block-correlated features with ({0.8,0.99}), zero-variance columns,
constant or nearly constant responses, and user-supplied lambda grids
with ties or extreme spacing. On orthonormal Gaussian cases, compare
Gaussian-family coefficients against the soft-thresholding oracle. More
generally, assert decreasing lambda order, finite predictions, objective
consistency, and graceful handling of known edge classes recorded in the
official changelog. Avoid brittle assertions about the exact active set
in highly correlated settings unless the oracle is specifically designed
for that regime.
citeturn11search1turn36search0turn35search0turn35search1

For `mlxs_cv_glmnet`, the key oracle is *reproducible cross-validation
geometry*. Use `nfolds` in ({3,5,10}), always test with explicit
`foldid`, and construct both benign stratified folds and adversarial
folds with missing classes or tiny test partitions. Assert that the same
data, same `foldid` and same seed produce the same `cvm`, `lambda.min`
and `lambda.1se`; that row permutation combined with the same fold
labels permuted produces identical outputs; that selected lambdas come
from valid locations in the fitted path; and that the one-standard-error
rule selects a penalty at least as large as the minimum-risk penalty in
the package’s ordering convention. Many operational bugs here are
leakage or determinism bugs, not numerical-optimisation bugs.
citeturn35search0turn13search7turn34search1turn34search7

For `mlxs_bootstrap`, separate the **resampling engine** from the
**statistic**. First test the engine: sample size preserved, replacement
really used, seed reproducibility, strata or clusters respected if
supported. Then test statistics: mean, median, OLS slope and one GLM
coefficient are enough to give broad coverage. Use (B {50,100}) for
PR-time smoke runs, ({999,1999}) nightly, and larger only for release
studies. For statistical validation, use normal, log-normal and
heavy-tailed inputs, and assess percentile/basic/BCa interval coverage
where implemented. Degenerate cases, such as all-identical observations
or statistics with zero variance, should produce collapsed but valid
intervals or clear errors, never reversed endpoints or NaNs.
citeturn29search7turn21search3turn21search10

### Example R pseudocode

``` r

gen_design <- function(n, p, rho = 0, tail = "gaussian", rank_def = FALSE, seed = 1) {
  set.seed(seed)

  # Correlated Gaussian design via AR(1) covariance
  Sigma <- outer(seq_len(p), seq_len(p), function(i, j) rho ^ abs(i - j))
  Z <- matrix(rnorm(n * p), n, p)
  X <- Z %*% chol(Sigma)

  if (tail == "t3") {
    X <- X / sqrt(rchisq(n, df = 3) / 3)
  } else if (tail == "lognormal") {
    X <- scale(exp(X), center = TRUE, scale = FALSE)
  }

  if (rank_def && p >= 3) {
    X[, p] <- X[, 1] + X[, 2]   # exact linear dependence
  }

  X
}

test_lm_properties <- function() {
  X <- gen_design(n = 64, p = 8, rho = 0.95, rank_def = TRUE, seed = 17)
  beta <- c(1, -1, rep(0, ncol(X) - 2))
  y <- drop(X %*% beta + rnorm(nrow(X), sd = 0.5))

  fit1 <- mlxs_lm(X, y)
  fit2 <- mlxs_lm(X[sample(nrow(X)), ], y[sample(nrow(X))])  # same permutation in real test

  # Assert on fitted values or predictions, not raw coefficients, when rank-deficient
  stopifnot(all(is.finite(predict(fit1, X))))
}
```

The key ideas in the R harness are to generate structured families
rather than ad hoc data, to test fitted values instead of raw
coefficients in non-identifiable problems, and to promote every
discovered failure into a permanent regression test under `tests/`. That
approach aligns with the documented testing patterns of R packages and
with simulation-study guidance.
citeturn4search4turn2search11turn16search2turn38search0turn38search1

### Example Python pseudocode

``` python
import numpy as np

def ar1_cov(p, rho):
    idx = np.arange(p)
    return rho ** np.abs(idx[:, None] - idx[None, :])

def gen_glm_case(n=200, p=20, rho=0.9, family="binomial", separable=False, seed=1):
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(ar1_cov(p, rho))
    X = rng.normal(size=(n, p)) @ L.T
    beta = np.zeros(p)
    beta[:3] = [2.0, -1.5, 1.0]

    eta = X @ beta
    if separable and family == "binomial":
        eta *= 8.0

    if family == "binomial":
        prob = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        y = rng.binomial(1, prob)
    elif family == "poisson":
        mu = np.exp(np.clip(eta, -10, 10))
        y = rng.poisson(mu)
    else:
        y = eta + rng.normal(scale=1.0, size=n)

    return X, y

def test_cv_glmnet_reproducibility():
    X, y = gen_glm_case(seed=314159)
    foldid = np.arange(len(y)) % 5

    fit_a = mlxs_cv_glmnet(X, y, foldid=foldid, seed=2718)
    fit_b = mlxs_cv_glmnet(X, y, foldid=foldid, seed=2718)

    assert fit_a.lambda_min == fit_b.lambda_min
    assert np.allclose(fit_a.cvm, fit_b.cvm)

def test_pca_subspace():
    rng = np.random.default_rng(17)
    U = rng.normal(size=(100, 3))
    V = rng.normal(size=(3, 20))
    X = U @ V  # exact rank-3
    pca = mlxs_pca(X, n_components=3)

    # Subspace and reconstruction, not sign-sensitive loadings
    Xhat = pca.transform(X) @ pca.components_ + pca.mean_
    assert np.linalg.norm(X - Xhat) / np.linalg.norm(X) < 1e-10
```

For Python-side tests, borrow two proven ideas from mature ecosystems:
reusable contract tests for estimator-like APIs, and explicit global
seed control for deterministic test runs. Those practices are prominent
in `scikit-learn` and `statsmodels`.
citeturn24search0turn24search4turn6search1turn2search13

## Prioritised CI pipeline

A high-value CI design for a statistical library is tiered, not flat.
Fast jobs should answer “did we break the contract?”. Nightly jobs
should answer “did we break robust behaviour on interesting inputs?”.
Weekly or release jobs should answer “are the advertised operating
characteristics still true within Monte Carlo uncertainty?”. This
separation mirrors the way mature libraries divide CI responsibilities
and is the only realistic way to balance speed and completeness.
citeturn7search3turn10search3turn13search1turn3search0

### Recommended tiers

| Tier | Trigger | Runtime target | Test content | Pass criteria |
|----|----|---:|----|----|
| Smoke | Every PR / commit | 2–5 min | API validation, deterministic metamorphic checks, NIST small cases, seed reproducibility, known regression fixtures | No crashes; exact warning/error contracts; tight deterministic tolerances |
| Core fuzz | Nightly | 20–45 min | Stratified fuzzing over structured generators, differential checks on small problems, optimisation diagnostics, fold/reproducibility tests | No new counterexamples; finite outputs; objective and prediction agreements within tolerances |
| Statistical validation | Weekly / release | 2–8 h | Monte Carlo bias/coverage/size/power studies, broader seed banks, extreme conditioning/stress sweeps | Operating characteristics inside predeclared confidence bands accounting for Monte Carlo error |
| Compatibility | Weekly / release | separate matrix | Multiple BLAS/LAPACK backends, Python/R versions, sparse/dense formats, single/multi-thread | No semantic regressions outside tolerance windows |

The triage rule should be simple. If a failure is fast, deterministic
and shrinks to a small case, it becomes a permanent regression test. If
it is statistical, preserve the seed, DGP parameters and summary
outputs, and schedule it in the nightly or weekly tier with an explicit
precision budget. If it is backend-specific, keep a compatibility
fixture in the matrix where it first appeared. This is much more
effective than merely increasing the number of random examples.
citeturn17search2turn18search5turn21search10turn10search2turn4search1

### Suggested prioritisation logic

Use a simple risk score for scheduling: \[ = / \]

In practice, this means: run `glm`, `cv_glmnet`, and any changed
optimisation code first; oversample separability, collinearity and
fold-geometry cases; and downgrade broad IID exploration when the queue
is tight. Historical issue trackers strongly suggest that grouped CV
behaviour, rank deficiency, convergence diagnostics and numerically
extreme inputs are where runtime buys the most safety.
citeturn34search1turn34search7turn26search5turn26search4turn10search1turn13search7turn35search0

### Mermaid flowchart

``` mermaid
flowchart TD
    A[Commit or scheduled run] --> B[Select changed modules and risk strata]
    B --> C[Smoke tier]
    C -->|pass| D[Core fuzz tier]
    C -->|fail| X[Shrink failing case]
    D -->|pass nightly| E[Statistical validation tier]
    D -->|fail| X
    E -->|pass| F[Release candidate evidence bundle]
    E -->|fail| X
    X --> Y[Store seed, DGP params, outputs, warnings]
    Y --> Z[Promote to permanent regression fixture]
    Z --> C
```

The most important automation detail is evidence capture. Every failure
record should store: seed, generator family, generator parameters, input
hashes, platform, backend, warning stream, optimiser diagnostics, and
the minimal shrunk input if available. Without that, fuzzing adds noise;
with it, fuzzing becomes a regression-test factory.
citeturn17search3turn17search4turn19search1

## Open questions and limitations

Some parts of the ecosystem are easier to inspect than others. `glmnet`
and `MASS` are visible mainly through official mirrors and changelogs
rather than rich public issue trackers, so the survey evidence for their
historical failures is necessarily more dependent on official release
notes than on issue discussions.
citeturn35search1turn35search0turn31search0

The precise expected behaviour of your `mlxs_*` functions may differ
from the canonical routines named `lm`, `glm`, `pca`, `glmnet`,
`cv_glmnet`, and `bootstrap`. If so, keep the same *stress families* but
revise the oracles to match your semantics. In particular, if you
provide robust covariance estimates, grouped resampling, custom
missing-value handling, or non-standard convergence criteria, those
should become first-class properties in the test matrix rather than
afterthoughts.
citeturn36search0turn8search6turn37search4turn29search7

The single most important implementation decision is not which specific
fuzzer you use. It is whether every test has a declared oracle, a
declared tolerance, and a declared runtime tier. Statistical software
fails most dangerously when it returns a plausible-looking answer for a
case it should have warned about, rejected, or treated differently. Your
fuzzing strategy should therefore be built not just to catch crashes,
but to catch *quiet statistical wrongness*.
citeturn28search1turn28search6turn19search1turn27search1turn21search10
