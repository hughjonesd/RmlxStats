# Notes from the first fuzz tests

## glm deterministic

- poisson has much larger fitted error than the others, esp large n and
  overdispersed
- huge differences in max_vcov_error but they are all tiny, 1e-6 or less
- big deviance errors

## glm monte carlo

- binomial all seems biased up

- poisson x1 biased down below 2 x mcse. Chance? ggplot(gmf,
  aes(y=family, x = bias, color = coefficient)) +
  geom_pointrange(aes(xmin = bias - 1.96\* mcse_bias, xmax = bias +
  1.96\*mcse_bias), position = position_dodge(width = 0.15)) +
  geom_vline(xintercept = 0) + theme_minimal()

- ci coverage looks pretty ok: ggplot(gmf, aes(y=family, x =
  ci_coverage, color = coefficient)) + geom_pointrange(aes(xmin =
  ci_coverage - 1.96\* mcse_coverage, xmax = ci_coverage +
  1.96\*mcse_coverage), position = position_dodge(width = 0.15)) +
  geom_vline(xintercept = 0.95) + theme_minimal()
