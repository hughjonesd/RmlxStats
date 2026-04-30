#' Shared mlxs model methods
#'
#' Methods for behavior shared by `mlxs_lm` and `mlxs_glm` through their
#' `mlxs_model` superclass.
#'
#' @param object An `mlxs_model` fit.
#' @param ... Additional arguments passed to underlying methods.
#' @param evaluate Logical; evaluate the updated call?
#'
#' @name mlxs-model-methods
#' @importFrom stats update.default
NULL

#' @export
#' @rdname mlxs-model-methods
update.mlxs_model <- function(object, ..., evaluate = TRUE) {
  update.default(object, ..., evaluate = evaluate)
}
