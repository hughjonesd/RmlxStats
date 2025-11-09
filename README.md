# RmlxStats

Statistical modelling front-ends that run on Apple GPU hardware via the [Rmlx](https://github.com/hughjonesd/Rmlx) array library.

## When to use mlxs_(g)lm

- GPU acceleration shines once the design matrix is large (tens of thousands of rows/columns) or you need to refit many times (e.g., bootstraps or cross-validation). In these cases the MLX QR/solve path can be several times faster than repeated `lm()`/`glm()` fits.
- For small problems (think classic textbook data) the overhead of launching GPU kernels and shuttling data usually outweighs any benefit; base R's CPU solvers will be faster.
- Residual bootstraps for Gaussian models are particularly effective: we reuse the original MLX QR factorisation and only resample residuals, so each replicate avoids a fresh factorisation.

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
