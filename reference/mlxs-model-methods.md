# Shared mlxs model methods

Methods for behavior shared by `mlxs_lm` and `mlxs_glm` through their
`mlxs_model` superclass.

## Usage

``` r
# S3 method for class 'mlxs_model'
update(object, ..., evaluate = TRUE)
```

## Arguments

- object:

  An `mlxs_model` fit.

- ...:

  Additional arguments passed to underlying methods.

- evaluate:

  Logical; evaluate the updated call?
