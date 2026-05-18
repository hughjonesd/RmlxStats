
skip_if_not_installed("sandwich")
requireNamespace("sandwich")

test_that("sandwich::estfun gives same results for lm and mlxs_lm", {
  mlxs_fit <- mlxs_lm(mpg ~ gear + am, mtcars)
  base_fit <- lm(mpg ~ gear + am, mtcars)
  
  expect_equal(as.matrix(sandwich::estfun(mlxs_fit)), 
               sandwich::estfun(base_fit),
               tolerance = 1e-6)
})

test_that("sandwich::estfun gives same results for glm and mlxs_glm", {
  mlxs_fit <- mlxs_glm(mpg ~ gear + am, mtcars, family = mlxs_gaussian())
  base_fit <- glm(mpg ~ gear + am, mtcars, family = "gaussian")
  
  expect_equal(as.matrix(sandwich::estfun(mlxs_fit)), 
               sandwich::estfun(base_fit),
               tolerance = 1e-6)
})

test_that("sandwich::bread gives same results for lm and mlxs_lm", {
  mlxs_fit <- mlxs_lm(mpg ~ gear + am, mtcars)
  base_fit <- lm(mpg ~ gear + am, mtcars)
  
  expect_equal(as.matrix(sandwich::bread(mlxs_fit)), 
               sandwich::bread(base_fit),
               tolerance = 1e-6)
})

test_that("sandwich::bread gives same results for glm and mlxs_glm", {
  mlxs_fit <- mlxs_glm(mpg ~ gear + am, mtcars, family = mlxs_gaussian())
  base_fit <- glm(mpg ~ gear + am, mtcars, family = "gaussian")
  
  expect_equal(as.matrix(sandwich::bread(mlxs_fit)), 
               sandwich::bread(base_fit),
               tolerance = 1e-6)
})

test_that("sandwich::vcovHC gives same results for lm and mlxs_lm", {
  skip_if_not_installed("sandwich")
  requireNamespace("sandwich")
  mlxs_fit <- mlxs_lm(mpg ~ gear + am, mtcars)
  base_fit <- lm(mpg ~ gear + am, mtcars)
  
  expect_equal(as.matrix(sandwich::vcovHC(mlxs_fit)), 
               sandwich::vcovHC(base_fit),
               tolerance = 1e-5)
})

test_that("sandwich::vcovHC gives same results for glm and mlxs_glm", {
  skip_if_not_installed("sandwich")
  requireNamespace("sandwich")
  mlxs_fit <- mlxs_glm(mpg ~ gear + am, mtcars, family = mlxs_gaussian())
  base_fit <- glm(mpg ~ gear + am, mtcars, family = "gaussian")
  
  expect_equal(as.matrix(sandwich::vcovHC(mlxs_fit)), 
               sandwich::vcovHC(base_fit),
               tolerance = 1e-5)
})

test_that("sandwich::vcovHAC gives same results for lm and mlxs_lm", {
  mlxs_fit <- mlxs_lm(mpg ~ gear + am, mtcars)
  base_fit <- lm(mpg ~ gear + am, mtcars)
  
  expect_equal(as.matrix(sandwich::vcovHAC(mlxs_fit)), 
               sandwich::vcovHAC(base_fit),
               tolerance = 1e-4)
})

test_that("sandwich::vcovHAC gives same results for glm and mlxs_glm", {
  mlxs_fit <- mlxs_glm(mpg ~ gear + am, mtcars, family = mlxs_gaussian())
  base_fit <- glm(mpg ~ gear + am, mtcars, family = "gaussian")
  
  expect_equal(as.matrix(sandwich::vcovHAC(mlxs_fit)), 
               sandwich::vcovHAC(base_fit),
               tolerance = 1e-4)
})


test_that("sandwich::vcovCL gives same results for lm and mlxs_lm", {
  # skip("FIXME: not yet within reasonable tolerance")
  mlxs_fit <- mlxs_lm(mpg ~ gear + am, mtcars)
  base_fit <- lm(mpg ~ gear + am, mtcars)
  
  expect_equal(as.matrix(sandwich::vcovCL(mlxs_fit, type = "HC0")), 
               sandwich::vcovCL(base_fit, type = "HC0"),
               tolerance = 1e-4)
  
  expect_equal(as.matrix(sandwich::vcovCL(mlxs_fit, ~cyl, type = "HC0")), 
               sandwich::vcovCL(base_fit, ~cyl, type = "HC0"),
               tolerance = 1e-4)
})

test_that("lmtest::coeftest gives same results for lm and mlxs_lm", {
  skip_if_not_installed("lmtest")
  mlxs_fit <- mlxs_lm(mpg ~ gear + am, mtcars)
  base_fit <- lm(mpg ~ gear + am, mtcars)
  
  expect_equal(
    lmtest::coeftest(base_fit, vcov. = sandwich::vcovHC),
    lmtest::coeftest(mlxs_fit, vcov. = as.matrix(sandwich::vcovHC(mlxs_fit))),
    tolerance = 1e-6,
    ignore_attr = TRUE
  )
})
