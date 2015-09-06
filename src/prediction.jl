# Abstract types

abstract PredictionModel{NDIn, NDOut}

### Generic functions

isvalidparam(pm::PredictionModel, θ::AbstractArray) = (size(θ) == paramsize(pm))


### Prediciton models

## LinearPred

immutable LinearPred <: PredictionModel{1,0}
    dim::Int
    LinearPred(d::Int) = new(d)
end

inputlen(pm::LinearPred) = pm.dim
inputsize(pm::LinearPred) = (pm.dim,)
outputlen(pm::LinearPred) = 1
outputsize(pm::LinearPred) = ()
paramlen(pm::LinearPred) = pm.dim
paramsize(pm::LinearPred) = (pm.dim,)

function ninputs(pm::LinearPred, x::StridedVecOrMat)
    @_checkdims inputlen(pm) == size(x,1)
    size(x, 2)
end

function predict{T<:BlasReal}(pm::LinearPred, θ::StridedVector{T}, x::StridedVector{T})
    @_checkdims length(θ) == length(x) == pm.dim
    dot(θ, x)
end

function predict{T<:BlasReal}(pm::LinearPred, θ::StridedVector{T}, X::StridedMatrix{T})
    @_checkdims length(θ) == size(X,1) == pm.dim
    X'θ
end

function predict!{T<:BlasReal}(r::StridedVector{T}, pm::LinearPred, θ::StridedVector{T}, X::StridedMatrix{T})
    d = pm.dim
    n = size(X,2)
    @_checkdims length(θ) == size(X,1) == pm.dim && n == length(r)
    At_mul_B!(r, X, θ)
end


## AffinePred

immutable AffinePred <: PredictionModel{1,0}
    dim::Int
    bias::Float64

    AffinePred(d::Int) = new(d, 1.0)
    AffinePred(d::Int, b::Real) = new(d, convert(Float64, b))
end

inputlen(pm::AffinePred) = pm.dim
inputsize(pm::AffinePred) = (pm.dim,)
outputlen(pm::AffinePred) = 1
outputsize(pm::AffinePred) = ()
paramlen(pm::AffinePred) = pm.dim + 1
paramsize(pm::AffinePred) = (pm.dim + 1,)

function ninputs(pm::AffinePred, x::StridedVecOrMat)
    @_checkdims inputlen(pm) == size(x,1)
    size(x, 2)
end

function predict{T<:BlasReal}(pm::AffinePred, θ::StridedVector{T}, x::StridedVector{T})
    d = pm.dim
    @_checkdims length(θ) == d + 1 && length(x) == d
    w = view(θ, 1:d)
    b = convert(T, pm.bias) * θ[d+1]
    dot(w, x) + b
end

function predict{T<:BlasReal}(pm::AffinePred, θ::StridedVector{T}, X::StridedMatrix{T})
    d = pm.dim
    @_checkdims length(θ) == d + 1 && size(X,1) == d
    w = view(θ, 1:d)
    b = convert(T, pm.bias) * θ[d+1]
    r = X'w;
    broadcast!(+, r, r, b)
end

function predict!{T<:BlasReal}(r::StridedVector{T}, pm::AffinePred, θ::StridedVector{T}, X::StridedMatrix{T})
    d = pm.dim
    n = size(X,2)
    @_checkdims length(θ) == d + 1 && size(X,1) == d && n == length(r)
    b = convert(T, pm.bias) * θ[d+1]
    w = view(θ, 1:d)
    At_mul_B!(r, X, w)
    broadcast!(+, r, r, b)
end


## MvLinearPred

immutable MvLinearPred <: PredictionModel{1,1}
    dim::Int
    k::Int

    MvLinearPred(d::Int, k::Int) = new(d, k)
end

inputlen(pm::MvLinearPred) = pm.dim
inputsize(pm::MvLinearPred) = (pm.dim,)
outputlen(pm::MvLinearPred) = pm.k
outputsize(pm::MvLinearPred) = (pm.k,)
paramlen(pm::MvLinearPred) = pm.k * pm.dim
paramsize(pm::MvLinearPred) = (pm.k, pm.dim)

function ninputs(pm::MvLinearPred, x::StridedVecOrMat)
    @_checkdims inputlen(pm) == size(x,1)
    size(x, 2)
end

function predict{T<:BlasReal}(pm::MvLinearPred, θ::StridedMatrix{T}, x::StridedVector{T})
    d = pm.dim
    k = pm.k
    @_checkdims size(θ) == (k,d) && length(x) == d
    θ * x
end

function predict{T<:BlasReal}(pm::MvLinearPred, θ::StridedMatrix{T}, X::StridedMatrix{T})
    d = pm.dim
    k = pm.k
    @_checkdims size(θ) == (k,d) && size(X,1) == d
    θ * X
end

function predict!{T<:BlasReal}(r::StridedMatrix{T}, pm::MvLinearPred, θ::StridedMatrix{T}, X::StridedMatrix{T})
    d = pm.dim
    k = pm.k
    n = size(X,2)
    @_checkdims size(θ) == (k,d) && size(X,1) == d && size(r) == (k,n)
    A_mul_B!(r, θ, X)
end


## MvAffinePred

immutable MvAffinePred <: PredictionModel{1,1}
    dim::Int
    k::Int
    bias::Float64

    MvAffinePred(d::Int, k::Int) = new(d, k, 1.0)
    MvAffinePred(d::Int, k::Int, b::Real) = new(d, k, convert(Float64, b))
end

inputlen(pm::MvAffinePred) = pm.dim
inputsize(pm::MvAffinePred) = (pm.dim,)
outputlen(pm::MvAffinePred) = pm.k
outputsize(pm::MvAffinePred) = (pm.k,)
paramlen(pm::MvAffinePred) = pm.k * (pm.dim + 1)
paramsize(pm::MvAffinePred) = (pm.k, pm.dim + 1)

function ninputs(pm::MvAffinePred, x::StridedVecOrMat)
    @_checkdims inputlen(pm) == size(x,1)
    size(x, 2)
end

function predict{T<:BlasReal}(pm::MvAffinePred, θ::StridedMatrix{T}, x::StridedVector{T})
    d = pm.dim
    k = pm.k
    @_checkdims size(θ) == (k,d+1) && length(x) == d
    W = view(θ, :, 1:d)
    b = view(θ, :, d+1)
    r = W * x
    axpy!(convert(T, pm.bias), b, r)
    r
end

function predict{T<:BlasReal}(pm::MvAffinePred, θ::StridedMatrix{T}, X::StridedMatrix{T})
    d = pm.dim
    k = pm.k
    @_checkdims size(θ) == (k,d+1) && size(X,1) == d
    W = view(θ, :, 1:d)
    b = view(θ, :, d+1)
    R = W * X
    bias = convert(T, pm.bias)
    for i = 1:size(X,2)
        axpy!(bias, b, view(R,:,i))
    end
    R
end

function predict!{T<:BlasReal}(r::StridedMatrix{T}, pm::MvAffinePred, θ::StridedMatrix{T}, X::StridedMatrix{T})
    d = pm.dim
    k = pm.k
    n = size(X,2)
    @_checkdims size(θ) == (k,d+1) && size(X,1) == d && size(r) == (k,n)
    bias = convert(T, pm.bias)
    W = view(θ, :, 1:d)
    b = view(θ, :, d+1)
    A_mul_B!(r, W, X)
    bias = convert(T, pm.bias)
    for i = 1:size(X,2)
        axpy!(bias, b, view(r,:,i))
    end
    r
end
