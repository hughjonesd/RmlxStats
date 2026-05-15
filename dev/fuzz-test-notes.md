# Notes from the first fuzz tests

## things to plot history

lm-det: not sure?
lm-mc: monte carlo bias and ci coverage with SEs
glm-det: aic and fitted errors; 
glm-mc: mc bias and ci as for lm-mc; 
glmnet-det: prediction errors
cv-glmnet: out of fold predictions
prcomp-det: subspace and reconstruction error
prcomp-mc: same

## working on the fuzz-test-results.Rmd

* We have too many columns, it would be better to keep things simpler, but
  not a top priority right now
* ChatGPT suggested metamorphic tests where we rescale and-or recentre 
  (some) x variables
  and expect coefficients to be rescaled the same
* From the report "tolerances are much more meaningful on **standardised quantities**: 
  standardised coefficients, linear predictors, fitted probabilities, residual norms,
  deviance, objective values, and KKT residuals". Not sure what the KKT residuals
  are. But in general should we standardise coefficients, except when we are 
  trying to test a scaling scenario? (Do we already?)

  
### mlxs_lm

* Is there a reason not to calculate the MCSE of the bias of our standard
  errors in mlxs_lm monte carlo? DONE
* Is a deterministic fuzz test still a fuzz test? Asking for a friend.
* We have a huge N for the slow bootstrap SEs, but not for the fast mlxs_lm
  default. DONE
  - Anyway, why is the bootstrap so slow for these cases? 
* Report suggests gaps include: correlated predictors and near-rank-deficient designs 
  for lm/glm. But we have "near-collinear" in lm-det. And "near-rank-deficient". So?
* Report on large P: "Large tests should be fewer, but deliberately pathological 
  and representative of the regimes your users actually care about: \(N \gg P\), 
  \(P \gg N\), high correlation blocks, duplicate or nearly duplicate columns, 
  broad dynamic range, and near-separation."
  
#### Substantive issues

* Why is x1 coverage so bad for the Heteroskedastic / n 80 / p 4
  Monte Carlo? (ANS: probably because it breaks the standard assumption for
  correct s.e.s)
* Near Rank Deficient with error = 1e-6 gives very poor results
* Longley and Wamperl are also both bad

  
### mlxs_glm

* Report suggests: cases near **complete or quasi-separation**, rare outcomes,
  extreme offsets, extreme weights, and very unbalanced probabilities. But
  we have correlated and rare event. So extreme weights, unbalanced probs?
  DONE for near-complete separation
* Does coefficient RMSE against truth tell us anything without a comparison to 
  base glm? DONE

### mlxs_glmnet

* how should lambda affect precision versus recall? presumably higher penalty
  for non-zero coefs reduces recall and increases precision, up to a point.
* What's the point of m.c. tests with just 5 reps in the fast path? Or 
  indeed just 20 in the full case? DONE
  - and why is lambda_index 6.4? Seems weird? And why not look at predictions
    rather than active set recovery? DONE
* Do we indeed lock down the lambda path/fold ids for these comparisons? DONE

  

#### Substantive issues

* binomial cases with Null Signal or AR1 Correlated, seems like later
  lambdas of mlxs_glmnet perform worse.


#### Research report comments:

* "For `glmnet`, cases where support is unstable, where multiple predictors 
  are highly correlated, and where the chosen \(\lambda\) is near a support 
  transition. The `glmnet` docs and papers explicitly note pathwise fitting,
  warm starts, and screening/strong rules; the strong-rules paper also discusses
  violations, which is exactly why KKT checks matter." What are KKT checks?
  - in particular, "chosen lambda near a support transition..."" how could we 
  pick one such? SKIPPED: too technical.
* "For `cv.glmnet`, exact equality is especially brittle unless you lock down 
  **`foldid` and the lambda grid**. The official docs state that by default the 
  full model gets a master lambda sequence and each fold gets its own sequence, so 
  fuzz tests that compare raw selected `lambda.min` or `lambda.1se` values without 
  fixing those ingredients can fail or pass for the wrong reasons." Does this mean us?
*  "`glmnet` is pathwise, uses warm starts, exploits sparsity, and 
  cross-validation can use fold-specific lambda sequences. In that regime, 
  large-\(P\) tests should explicitly watch the things that small toy tests 
  miss: active-set stability, KKT residuals, objective monotonicity along the 
  path, support changes under near-duplicate columns, and reproducibility under 
  fixed folds."
  
### mlxs_prcomp


* We should test more high N/high P cases and work the randomized path.

* "you should not compare raw eigenvectors elementwise except in contrived cases. 
  The official docs for `PCA` and `randomized_svd` emphasise solver changes by
  size/shape and note that randomized methods require oversampling and 
  power-iteration choices for slow spectral decay and noisy problems. The right 
  oracle is **subspace agreement**, explained variance, singular values, 
  reconstruction error, and invariance to column sign flips, not 
  exact vector orientation." OK fine. invariance to colum sign flips would be
  a nice metamorphic test if we don't?
  
#### Substantive issues

* Sparse Dense Randomized is problematic, very; so is Near Duplicate Exact
  for subspace error.
  
## glm deterministic

- poisson has much larger fitted error than the others, esp large n and
overdispersed
- huge differences in max_vcov_error but they are all tiny, 1e-6 or less
- big deviance errors

## glm monte carlo

- binomial all seems biased up
- poisson x1 biased down below 2 x mcse. Chance?
ggplot(gmf, aes(y=family, x = bias, color = coefficient)) + 
  geom_pointrange(aes(xmin = bias - 1.96* mcse_bias, 
                      xmax = bias + 1.96*mcse_bias), 
                  position = position_dodge(width = 0.15)) + 
  geom_vline(xintercept = 0) + 
  theme_minimal()
  
- ci coverage looks pretty ok:
ggplot(gmf, aes(y=family, x = ci_coverage, color = coefficient)) + 
  geom_pointrange(aes(xmin = ci_coverage - 1.96* mcse_coverage, 
                      xmax = ci_coverage + 1.96*mcse_coverage), 
                  position = position_dodge(width = 0.15)) + 
  geom_vline(xintercept = 0.95) + 
  theme_minimal()