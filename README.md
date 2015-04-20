# EmpiricalRisks

This Julia package provides a collection of predictors and loss functions, mainly to support the implementation of (regularized) empirical risk minimization methods.

**Test Status:**
[![Build Status](https://travis-ci.org/lindahua/EmpiricalRisks.jl.svg?branch=master)](https://travis-ci.org/lindahua/EmpiricalRisks.jl)
[![EmpiricalRisks](http://pkg.julialang.org/badges/EmpiricalRisks_release.svg)](http://pkg.julialang.org/?pkg=EmpiricalRisks&ver=release)

## Overview

This package provides basic components for implementing regularized empirical risk minimization:

![regerm](imgs/regerm.png)

- **Prediction models** ``u = f(x; θ)``

  - [ ] linear prediction
  - [ ] affine prediction
  - [ ] multivariate linear prediction
  - [ ] multivariate affine prediction

- **Loss functions** ``loss(u, y)``

  - [ ] squared loss
  - [ ] absolute loss
  - [ ] huber loss
  - [ ] hinge loss
  - [ ] smoothed hinge loss
  - [ ] logistic loss
  - [ ] sum squared loss (for multivariate prediction)
  - [ ] multinomial logistic loss

  **Notes:**

  - For each loss function, we provide methods to compute the loss value, the derivative/gradient, or both (at the same time).
  - For each (consistent) combination of loss function and prediction model (which together are referred to as a *risk model*), we provide methods to compute the total risk and the gradient *w.r.t.* the parameter.


- **Regularizers**  

  - [ ] squared L2
  - [ ] L1
  - [ ] elastic net (L1 + squared L2)

  **Notes:**

  - For each regularizer, we provide methods to evaluate the regularization value, the gradient, and the proximal operator.


**Remarks:** All functions in this package are carefully optimized and tested.


## Documentation

Here is the [Detailed Documentation](http://empiricalrisksjl.readthedocs.org/en/latest/).
[![Documentation Status](https://readthedocs.org/projects/empiricalrisksjl/badge/?version=latest)](https://readthedocs.org/projects/empiricalrisksjl/?badge=latest)
