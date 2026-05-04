#' Re-export generics
#'
#' These generics are re-exported from the generics package for convenience.
#'
#' @name generics-reexports
#' @param x,... Passed to the generic.
NULL

#' @importFrom generics tidy
#' @export
#' @rdname generics-reexports
generics::tidy

#' @importFrom generics glance
#' @export
#' @rdname generics-reexports
generics::glance

#' @importFrom generics augment
#' @export
#' @rdname generics-reexports
generics::augment

# Adding generics::estfun here gave an R CMD CHECK warning, so don't do that
