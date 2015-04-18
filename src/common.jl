
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

macro _checkdims(cond)
    quote
        ($cond) || throw(DimensionMismatch())
    end
end

function axpby!{T<:BlasReal}(a::T, x::StridedVector{T}, b::T, y::StridedVector{T})
    if b == zero(T)
        scale!(y, x, a)
    elseif b == one(T)
        axpy!(a, x, y)
    else
        n = length(x)
        length(y) == n || throw(DimensionMismatch())
        @inbounds for i = 1:n
            y[i] = a * x[i] + b * y[i]
        end
    end
    y
end
