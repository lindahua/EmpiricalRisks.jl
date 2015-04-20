Welcome to EmpiricalRisks's documentation!
==========================================


This package provides the basic components for *(regularized) empirical risk minization*, which is generally formulated as follows

.. math::

    \sum_{i=1}^n loss(f(x_i; \theta), y_i) + r(\theta)

As we can see, this formulation involves several components:

- **Prediction model:** :math:`f(x; \theta)`, which takes an input :math:`x` and a parameter :math:`\theta` and produces an output (say :math:`u`).
- **Loss function:** :math:`loss(u, y)`, which compares the predicted output :math:`u` and a desired response :math:`y`, and produces a real value that measuring the *loss*. Generally, better prediction yields smaller loss.
- **Risk model:** :math:`loss(f(x; \theta), y)`, the *prediction model* and the *loss* together are referred to as the *risk model*. When the data `x` and `y` are given, the risk model can be considered as a function of `theta`.
- **Regularizer:**  :math:`r(\theta)` is often introduced to regularize the parameter, which, when used properly, can improve the numerical stability of the problem and the generalization performance of the estimated model.

This package provides these components, as well as the gradient computation routines and proximal operators, to support the implementation of various empirical risk minimization algorithms.

All functions in this packages are well optimized and systematically tested.


Contents:
----------

.. toctree::
   :maxdepth: 2

   prediction.rst
   loss.rst
   riskmodels.rst
