# Abstract types

abstract PredictionModel
abstract UnivariatePredictionModel <: PredictionModel
abstract MultivariatePredictionModel <: PredictionModel

### Generic functions

nsamples(::UnivariatePredictionModel, x::AbstractVector) = 1
nsamples(::UnivariatePredictionModel, X::AbstractMatrix) = size(X, 2)

nsamples(::MultivariatePredictionModel, x::AbstractVector) = 1
nsamples(::MultivariatePredictionModel, X::AbstractMatrix) = size(X, 2)


### Prediciton models

immutable LinearPred <: UnivariatePredictionModel
end

predict{T<:BlasReal}(::LinearPred, θ::StridedVector{T}, x::StridedVector{T}) = dot(θ, x)
predict{T<:BlasReal}(::LinearPred, θ::StridedVector{T}, X::StridedMatrix{T}) = X'θ


immutable AffinePred <: UnivariatePredictionModel
    bias::Float64

    AffinePred() = new(1.0)
    AffinePred(b::Real) = new(convert(Float64, b))
end

function predict{T<:BlasReal}(pred::AffinePred, θ::StridedVector{T}, x::StridedVector{T})
    d = length(x)
    length(θ) == d + 1 || throw(DimensionMismatch("Inconsistent input dimensions."))
    w = view(θ, 1:d)
    b = convert(T, pred.bias) * θ[d+1]
    dot(w, x) + b
end

function predict{T<:BlasReal}(pred::AffinePred, θ::StridedVector{T}, X::StridedMatrix{T})
    d = size(X,1)
    length(θ) == d + 1 || throw(DimensionMismatch("Inconsistent input dimensions."))
    w = view(θ, 1:d)
    b = convert(T, pred.bias) * θ[d+1]
    r = X'w;
    broadcast!(+, r, r, b)
end


immutable MvLinearPred <: MultivariatePredictionModel
end

predict{T<:BlasReal}(pred::MvLinearPred, θ::StridedMatrix{T}, x::StridedVector{T}) = θ'x
predict{T<:BlasReal}(pred::MvLinearPred, θ::StridedMatrix{T}, X::StridedMatrix{T}) = θ'X


immutable MvAffinePred <: MultivariatePredictionModel
    bias::Float64

    MvAffinePred() = new(1.0)
    MvAffinePred(b::Real) = new(convert(Float64, b))
end

function predict{T<:BlasReal}(pred::MvAffinePred, θ::StridedMatrix{T}, x::StridedVector{T})
    d = length(x)
    size(θ,1) == d + 1 || throw(DimensionMismatch("Inconsistent input dimensions."))
    w = view(θ, 1:d, :)
    b = scale(rowvec_view(θ, d+1), convert(T, pred.bias))
    r = w'x
    broadcast!(+, r, r, b)
end

function predict{T<:BlasReal}(pred::MvAffinePred, θ::StridedMatrix{T}, X::StridedMatrix{T})
    d = size(X,1)
    size(θ,1) == d + 1 || throw(DimensionMismatch("Inconsistent input dimensions."))
    w = view(θ, 1:d, :)
    b = scale(rowvec_view(θ, d+1), convert(T, pred.bias))
    r = w'X
    broadcast!(+, r, r, b)
end
