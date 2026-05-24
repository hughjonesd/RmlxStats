#' Shared model parameter documentation
#'
#' @param formula Model formula.
#' @param data Optional data frame, tibble, or environment containing the
#'   variables in the model.
#' @param subset Optional expression for subsetting observations.
#' @param weights Optional non-negative observation weights.
#' @param na.action How to handle missing values.
#' @param rank_tol Optional relative tolerance used to detect rank-deficient
#'   systems. `NULL` uses the package default, which varies by dtype and is
#'   1e-6 for float32 matrices. Set to `FALSE` to skip rank checks entirely. 
#'   Note that higher numbers indicate *lower* tolerance.
#' @keywords internal
#' @name mlxs_model_params
NULL
