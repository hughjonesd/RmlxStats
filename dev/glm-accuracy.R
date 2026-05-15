
library(tidyverse)

results <- lapply(10^-(4:16), function (eps) {
  fml <- cut=="Fair" ~ depth + carat + price
  data <- diamonds
  base <- glm(fml, data = data,  family = "binomial",
              control = list(epsilon = eps))
  m <- mlxs_glm(fml, data = data, family = "binomial",
                control = list(epsilon = eps))
  
  data.frame(eps, term = rep(names(coef(base)), 2),
             value = c(coef(base), as.numeric(coef(m))),
             model = rep(c("base", "mlxs"), each = length(coef(base))))
}) |> do.call(what = rbind) 
  
results <- results |> 
  mutate(.by = c(term, model),
         dvalue = c(NA, diff(value)),
         ) |> 
  mutate(.by = term,
    diff_ref = value - value[model == "base" & eps == 10^-16],
    diff_mods = rep(value[model=="base"] - value[model=="mlxs"], 2)
  )

ggplot(results, aes(eps, abs(diff_ref), color = model, 
                    shape = term)) + 
  geom_point() + 
  geom_line() + 
  scale_x_log10() +
  coord_cartesian(reverse = "x") +
  scale_y_log10() +
  theme_minimal()

results |> 
  filter(model == "base") |> 
  ggplot(aes(eps, abs(diff_mods), 
                    shape = term)) + 
    geom_point() + 
    geom_line() + 
    scale_x_log10() +
    coord_cartesian(reverse = "x") +
    scale_y_log10() +
    theme_minimal()

ggplot(results, aes(eps, abs(dvalue), color = model, shape = term)) + 
  geom_point() + 
  geom_line() + 
  scale_x_log10() +
  coord_cartesian(reverse = "x") +
  scale_y_log10() +
  theme_minimal()
