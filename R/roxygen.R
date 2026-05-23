#' Shared model parameter documentation
#'
#' @param formula Model formula.
#' @param data Optional data frame, tibble, or environment containing the
#'   variables in the model.
#' @param subset Optional expression for subsetting observations.
#' @param weights Optional non-negative observation weights.
#' @param na.action A function indicating how missing values should be handled.
#'   Defaults to [stats::na.exclude()] so residuals, fitted values, and
#'   training-data predictions are padded back to the original row count.
#' @param rank_tol Optional relative tolerance used to detect rank-deficient
#'   systems. `NULL` uses the package default, preserving the historical
#'   dtype-sensitive cutoff. Set to `FALSE` to skip rank checks entirely.
#' @keywords internal
#' @name mlxs_model_params
NULL
