
# adapted from sandwich:::estfun.lm
#' @exportS3Method sandwich::estfun
#' @rdname mlxs-lm-methods
estfun.mlxs_lm <- function(x, ..., output = c("matrix", "mlx")) {
  output <- match.arg(output)
  xmat <- model.matrix(x)
  xmat <- naresid(x$na.action, xmat)
  if (any(alias <- is.na(coef(x)))) 
    xmat <- xmat[, !alias, drop = FALSE]
  wts <- weights(x) %||% 1
  res <- residuals(x)
  wtd_res <- as.vector(res) * Rmlx::drop(wts)
  wtd_res <- Rmlx::mlx_reshape(wtd_res, c(length(wtd_res), 1L))
  rval <- wtd_res * xmat
  attr(rval, "assign") <- NULL
  attr(rval, "contrasts") <- NULL
  if (identical(output, "matrix")) rval <- Rmlx::as_r(rval)
  return(rval)
}



# adapted from sandwich:::estfun.glm
#' @exportS3Method sandwich::estfun
#' @rdname mlxs-glm-methods
estfun.mlxs_glm <- function(x, ..., output = c("matrix", "mlx")) {
  output <- match.arg(output)
  xmat <- model.matrix(x, )
  xmat <- naresid(x$na.action, xmat)
  if (any(alias <- is.na(coef(x)))) 
    xmat <- xmat[, !alias, drop = FALSE]
  wts <- weights(x, "working")
  wres <- residuals(x, "working") * wts
  dispersion <- dispersion(x)
  rval <- wres * xmat / dispersion
  attr(rval, "assign") <- NULL
  attr(rval, "contrasts") <- NULL
  if (identical(output, "matrix")) rval <- as.matrix(rval)
  return(rval)
}

# adapted from stats:::hatvalues
#' @export
#' @rdname mlxs-lm-methods
hatvalues.mlxs_lm <- function(model, ..., output = c("matrix", "mlx")) {
  output <- match.arg(output)
  Q <- model$qr$Q
  hv <- Rmlx::rowSums(Q^2)
  if (identical(output, "matrix")) hv <- Rmlx::as_r(hv)
  hv
}

#' @export
#' @rdname mlxs-glm-methods
hatvalues.mlxs_glm <- hatvalues.mlxs_lm 



#' @exportS3Method sandwich::bread
#' @rdname mlxs-lm-methods
bread.mlxs_lm <- function (x, ...) {
  summary(x)$cov.unscaled * nobs(x)
}

#' @exportS3Method sandwich::bread
#' @rdname mlxs-glm-methods
bread.mlxs_glm <- function (x, ...) {
  if (!is.null(x$na.action)) 
    class(x$na.action) <- "omit"
  sx <- summary(x)
  dispersion <- dispersion(x)
  return(sx$cov.unscaled * nobs(x) * dispersion)
}

dispersion <- function (x) {
  if (x$family$family %in% c("poisson", "binomial")) {
    1
  } else {
    wts <- weights(x, "working")
    wres <- residuals(x, "working") * wts
    sum(wres^2)/sum(wts)
  } 
}