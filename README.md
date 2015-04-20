# EmpiricalRisks

This Julia package provides a collection of predictors and loss functions, mainly to support the implementation of (regularized) empirical risk minimization methods.

[![Build Status](https://travis-ci.org/lindahua/EmpiricalRisks.jl.svg?branch=master)](https://travis-ci.org/lindahua/EmpiricalRisks.jl)
[![EmpiricalRisks](http://pkg.julialang.org/badges/EmpiricalRisks_release.svg)](http://pkg.julialang.org/?pkg=EmpiricalRisks&ver=release)

-----

## Overview

This package provides the basic components for *(regularized) empirical risk minization*, which is generally formulated as follows

![regerm](imgs/regerm.png)

As we can see, this formulation involves several components:

- **Prediction model:** ``f(x; θ)``, which takes an input `x` and a parameter `θ` and produces an output (say `u`).
- **Loss function:** ``loss(u, y)``, which compares the predicted output `u` and a desired response `y`, and produces a real value that measuring the *loss*. Generally, better prediction yields smaller loss.
- **Risk model:** ``loss(f(x; θ), y)``, the *prediction model* and the *loss* together are referred to as the *risk model*. When the data `x` and `y` are given, the risk model can be considered as a function of `theta`.
- **Regularizer:**  ``r(θ)`` is often introduced to regularize the parameter, which, when used properly, can improve the numerical stability of the problem and the generalization performance of the estimated model.

This package provides all these components, as well as the gradient computation routines and proximal operators, to support the implementation of various empirical risk minimization algorithms.

All functions in this packages are well optimized and systematically tested.


----

## Prediction models

A *prediction model* `f(x; θ)` is a function with two arguments: the input feature `x` and the predictor parameter `θ`. All prediction models are instances of an the abstract type ``PredictionModel``, defined as follows:

```julia
abstract PredictionModel{NDIn, NDOut}

# NDIn:  The number of dimensions of each input (0: scalar, 1: vector, 2: matrix, ...)
# NDOut: The number of dimensions of each output (0: scalar, 1: vector, 2: matrix, ...)
```

The package provides the following prediction models:

- **(Univariate) Linear Prediction:** `u = θ'x` (*vector -> scalar*)

```julia
immutable LinearPred <: PredictionModel{1,0}
    dim::Int
    LinearPred(d::Int) = new(d)
end
```

- **(Univariate) Affine Prediction:** `u = w'x + b * bias` with ``θ = [w; b]`` (*vector -> scalar*)

```julia
immutable AffinePred <: PredictionModel{1,0}
    dim::Int
    bias::Float64
    AffinePred(d::Int) = new(d, 1.0)
    AffinePred(d::Int, b::Real) = new(d, convert(Float64, b))
end
```

- **(Multivariate) Linear Prediction:** `u = W * x` where `θ = W` is of size ``(k, d)`` (*d-vector -> k-vector*)

```julia
immutable MvLinearPred <: PredictionModel{1,1}
    dim::Int
    k::Int
    MvLinearPred(d::Int, k::Int) = new(d, k)
end
```

- **(Multivariate) Affine Prediction:** `u = W * x + b * bias` where `θ = [W b]` is of size `(k, d+1)` (*d-vector -> k-vector*)

```julia
immutable MvAffinePred <: PredictionModel{1,1}
    dim::Int
    k::Int
    bias::Float64
    MvAffinePred(d::Int, k::Int) = new(d, k, 1.0)
    MvAffinePred(d::Int, k::Int, b::Real) = new(d, k, convert(Float64, b))
end
```

Each prediction model implements the following methods:

- **inputlen**(pm)

  Return the length of each input.

- **inputsize**(pm)

  Return the size of each input.

- **outputlen**(pm)

  Return the length of each output.

- **outputsize**(pm)

  Return the size of each output.

- **paramlen**(pm)

  Return the length of the parameter.

- **paramsize**(pm)

  Return the size of the parameter.

- **ninputs**(pm, x)

  Verify the validity of `x` as a single input or as a batch of inputs.
  If `x` is valid, it returns the number of inputs in array `x`, otherwise, it raises an error.

- **predict**(pm, θ, x)

  Predict the output given the parameter `θ` and the input `x`. 
