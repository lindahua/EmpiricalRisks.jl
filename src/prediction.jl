# Abstract types

abstract PredictionModel
abstract UnivariatePredictionModel <: PredictionModel
abstract MultivariatePredictionModel <: PredictionModel

### Generic functions

nsamples(::UnivariatePredictionModel, x::AbstractVector) = 1
nsamples(::UnivariatePredictionModel, X::AbstractMatrix) = size(X, 2)

nsamples(::MultivariatePredictionModel, x::AbstractVector) = 1
nsamples(::MultivariatePredictionModel, X::AbstractMatrix) = size(X, 2)

grad{T<:FloatingPoint}(pred::PredictionModel, θ::AbstractVector{T}, x::AbstractVector) =
    grad!(pred, Array(T, length(θ)), θ, x)

grad{T<:FloatingPoint}(pred::PredictionModel, θ::AbstractVector{T}, X::AbstractMatrix) =
    grad!(pred, Array(T, length(θ), size(X,2)), θ, X)

grad{T<:FloatingPoint}(pred::PredictionModel, θ::AbstractMatrix{T}, x::AbstractVector) =
    grad!(pred, Array(T, size(θ)), θ, x)

grad{T<:FloatingPoint}(pred::PredictionModel, θ::AbstractMatrix{T}, X::AbstractMatrix) =
    grad!(pred, Array(T, size(θ,1), size(θ,2), size(X,2)), θ, X)

total_grad{T<:FloatingPoint}(pred::PredictionModel, θ::AbstractArray{T}, X, c::StridedVector{T}) =
    total_grad!(pred, Array(T, size(θ)), θ, X, c)


### Prediciton models

# LinearPred

immutable LinearPred <: UnivariatePredictionModel
end

predict{T<:BlasReal}(::LinearPred, θ::StridedVector{T}, x::StridedVector{T}) = dot(θ, x)
predict{T<:BlasReal}(::LinearPred, θ::StridedVector{T}, X::StridedMatrix{T}) = X'θ

grad!{T<:BlasReal}(::LinearPred, g::StridedVector{T}, θ::StridedVector, x::StridedVector{T}) =
    copy!(g, x)

grad!{T<:BlasReal}(::LinearPred, g::StridedMatrix{T}, θ::StridedVector, X::StridedMatrix{T}) =
    copy!(g, X)

add_grad!{T<:BlasReal}(::LinearPred, g::StridedVector{T}, θ::StridedVector{T}, x::StridedVector{T}, c::Real) =
    axpy!(convert(T, c), x, g)

total_grad!{T<:BlasReal}(::LinearPred, g::StridedVector{T}, θ::StridedVector{T}, X::StridedMatrix{T}, c::StridedVector{T}) =
    gemv!('N', one(T), X, c, zero(T), g)

accum_grad!{T<:BlasReal}(::LinearPred, g::StridedVector{T}, θ::StridedVector{T}, X::StridedMatrix{T}, c::StridedVector{T}) =
    gemv!('N', one(T), X, c, one(T), g)


# AffinePred

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

function grad!{T<:BlasReal}(pred::AffinePred, g::StridedVector{T}, θ::StridedVector, x::StridedVector{T})
    d = length(x)
    length(g) == d+1 || throw(DimensionMismatch())
    copy!(view(g, 1:d), x)
    g[d+1] = convert(T, pred.bias)
    return g
end

function grad!{T<:BlasReal}(pred::AffinePred, g::StridedMatrix{T}, θ::StridedVector, X::StridedMatrix{T})
    d, n = size(X)
    size(g) == (d+1, n) || throw(DimensionMismatch())
    b = convert(T, pred.bias)
    for i = 1:n
        copy!(view(g, 1:d, i), view(X,:,i))
        g[d+1, i] = b
    end
    return g
end

function add_grad!{T<:BlasReal}(pred::AffinePred, g::StridedVector{T}, θ::StridedVector{T}, x::StridedVector{T}, c::Real)
    d = length(x)
    length(g) == d+1 || throw(DimensionMismatch())
    axpy!(convert(T, c), x, view(g, 1:d))
    g[d+1] += pred.bias * convert(T, c)
    return g
end

function total_grad!{T<:BlasReal}(pred::AffinePred, g::StridedVector{T}, θ::StridedVector{T}, X::StridedMatrix{T}, c::StridedVector{T})
    d = size(X, 1)
    length(g) == d+1 || throw(DimensionMismatch())
    gemv!('N', one(T), X, c, zero(T), view(g, 1:d))
    g[d+1] = pred.bias * sum(c)
    return g
end

function accum_grad!{T<:BlasReal}(pred::AffinePred, g::StridedVector{T}, θ::StridedVector{T}, X::StridedMatrix{T}, c::StridedVector{T})
    d = size(X, 1)
    length(g) == d+1 || throw(DimensionMismatch())
    gemv!('N', one(T), X, c, one(T), view(g, 1:d))
    g[d+1] += pred.bias * sum(c)
    return g
end


# MvLinearPred

immutable MvLinearPred <: MultivariatePredictionModel
end

predict{T<:BlasReal}(pred::MvLinearPred, θ::StridedMatrix{T}, x::StridedVector{T}) = θ'x
predict{T<:BlasReal}(pred::MvLinearPred, θ::StridedMatrix{T}, X::StridedMatrix{T}) = θ'X


# MvAffinePred

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
