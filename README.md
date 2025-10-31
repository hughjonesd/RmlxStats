# RmlxStats

Statistical modelling front-ends that run on Apple GPU hardware via the [Rmlx](https://github.com/hughjonesd/Rmlx) array library.

## Installation

1. Install Apple's MLX runtime (provides the Metal-backed tensor engine):
   ```bash
   brew install mlx
   ```
2. Install the development dependencies in R (requires R 4.5+ on Apple Silicon):
   ```r
   install.packages(c("devtools", "nycflights13", "bench", "fixest", "RcppEigen", "speedglm"))
   ```
3. Install Rmlx and RmlxStats from GitHub:
   ```r
   remotes::install_github("hughjonesd/Rmlx")
   remotes::install_github("hughjonesd/RmlxStats")
   ```
