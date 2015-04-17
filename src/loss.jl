
## Squared loss (for linear regression)
#
#   loss(p, y) := (1/2) * (p - y)^2
#
type SqrLoss <: UnivariateLoss
end

value{T<:BlasReal}(::SqrLoss, u::T, v::T) = half(abs2(u - v))
value_and_deriv{T<:BlasReal}(::SqrLoss, u::T, y::T) = (r = u - y; v = half(abs2(r)); (v, r))


## Hinge loss (for SVM)
#
#   loss(p, y) := max(1 - y * p, 0)
#
type HingeLoss <: UnivariateLoss
end

value{T<:BlasReal}(::HingeLoss, p::T, y::T) = nonneg(one(T) - y * p)
deriv{T<:BlasReal}(::HingeLoss, p::T, y::T) = y * p < one(T) ? -y : zero(T)

function value_and_deriv{T<:BlasReal}(::HingeLoss, p::T, y::T)
    yp = y * p
    yp >= one(T) ? (zero(T), zero(T)) : (one(T) - yp, -y)
end


## Logistic loss (for logistic regression)
#
#   loss(p, y) := log(1 + exp(-y * p))
#
type LogisticLoss <: UnivariateLoss
end

value{T<:BlasReal}(::LogisticLoss, p::T, y::T) =
    (yp = y * p; yp >= zero(T) ? log1p(exp(-yp)) : log1p(exp(yp)) - yp)

function deriv{T<:BlasReal}(::LogisticLoss, p::T, y::T)
    yp = y * p
    if yp >= zero(T)
        e = exp(-yp)
        -y * e / (one(T) + e)
    else
        -y / (one(T) + exp(yp))
    end
end

function value_and_deriv{T<:BlasReal}(::LogisticLoss, p::T, y::T)
    yp = y * p
    if yp >= zero(T)
        e = exp(-yp)
        (log1p(e), -y * e / (one(T) + e))
    else
        e = exp(yp)
        (log1p(e) - yp, -y / (one(T) + e))
    end
end


## general functions for multivariate loss

function value_and_grad{T<:BlasReal}(loss::MultivariateLoss, p::StridedVector{T}, y)
    g = zeros(T, length(p))
    value_and_grad!(loss, g, p, y)
end


## Multinomial logistic loss (for Multinomial logistic regression)
#
#   loss(p, y) := log(sum_k exp(p[k])) - p[y]
#

type MultiLogisticLoss <: MultivariateLoss
end

function value{T<:BlasReal}(::MultiLogisticLoss, p::StridedVector{T}, y::Integer)
    k = length(p)
    pmax = maximum(p)
    s = zero(T)
    @inbounds for i = 1:k
        s += exp(p[i] - pmax)
    end
    pmax + log(s) - p[y]
end

function grad!{T<:BlasReal}(::MultiLogisticLoss, g::StridedVector{T}, p::StridedVector{T}, y::Integer)
    k = length(p)
    length(g) == k || throw(DimensionMismatch("Inconsistent input dimensions."))
    pmax = maximum(p)
    s = zero(T)
    @inbounds for i = 1:k
        pi = exp(p[i] - pmax)
        g[i] = pi
        s += pi
    end
    @inbounds for i = 1:k
        g[i] /= s
    end
    g[y] -= one(T)
    g
end

function value_and_grad!{T<:BlasReal}(::MultiLogisticLoss, g::StridedVector{T}, p::StridedVector{T}, y::Integer)
    k = length(p)
    length(g) == k || throw(DimensionMismatch("Inconsistent input dimensions."))

    pmax = maximum(p)
    p_y = p[y]  # g and p can be the same array, so have to cache p[y] before p is overwritten

    s = zero(T)
    @inbounds for i = 1:k
        pi = exp(p[i] - pmax)
        g[i] = pi
        s += pi
    end

    @inbounds for i = 1:k
        g[i] /= s
    end
    g[y] -= one(T)
    return (pmax - p_y + log(s), g)
end
