library(ggplot2)
path <- "dev/lm-precision-benchmark"
before <- read.csv(file.path(path, "before-summary.csv"))
after <- read.csv(file.path(path, "after-summary.csv"))
precision_before <- read.csv(file.path(path, "before-precision.csv"))
precision_after <- read.csv(file.path(path, "after-precision.csv"))
before <- merge(before, precision_before)
after <- merge(after, precision_after)
before <- subset(before, scenario == "low_noise_correlated" &
                   method == "current")
after <- subset(after, scenario == "low_noise_correlated")
before$label <- "GPU before"
after$label <- c(current = "GPU corrected", cpu32 = "CPU float32",
                 float64 = "CPU float64")[after$method]
results <- rbind(before, after)
results$label <- factor(results$label,
                        c("GPU before", "GPU corrected", "CPU float32",
                          "CPU float64"))
p <- ggplot(results, aes(median * 1000, coef_error_se, color = label)) +
  geom_errorbar(aes(xmin = p10 * 1000, xmax = p90 * 1000),
                orientation = "y", width = 0, linewidth = 1) +
  geom_point(size = 4) +
  scale_x_log10(breaks = c(20, 50, 100, 200)) +
  scale_y_log10(breaks = c(1e-7, 1e-4, 0.1, 1, 100)) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "grey60") +
  scale_color_manual(values = c("#b84038", "#007d74", "#c88500", "#3461a5")) +
  labs(title = "Low-noise fit: much less error for a small time cost",
       subtitle = "200,000 rows, 51 columns; 30 timing rounds per method",
       x = "Median fit time (ms, logarithmic scale)",
       y = "Worst coefficient error / reference SE (logarithmic scale)",
       color = NULL,
       caption = "Horizontal ranges: timing percentiles 10–90. Lower and left are better.") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom", plot.title = element_text(face = "bold"))
ggsave(file.path(path, "tradeoff.png"), p, width = 9, height = 5.5, dpi = 160)
