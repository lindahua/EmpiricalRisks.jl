
# Abstract type hierarcgy

abstract Loss
abstract UnivariateLoss <: Loss
abstract MultivariateLoss <: Loss

abstract PredictionModel
abstract UnivariatePredictionModel <: PredictionModel
abstract MultivariatePredictionModel <: PredictionModel

# utilities

half(x::Float64) = x * 0.5
half(x::Float32) = x *0.5f0

nonneg(x::Float64) = x < 0.0 ? 0.0 : x  # this also properly takes care of NaN
nonneg(x::Float32) = x < 0.0f0 ? 0.0f0 : x