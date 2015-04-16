
## Squared loss (for linear regression)
#
#   loss(θ, x, y) := (1/2) * (θ'x - y)^2
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

function value_and_deriv{T<:BlasReal}(::LogisticLoss, p::T, y::T)
    yp = y * p
    if yp >= zero(T)
        e = exp(-yp)
        (log1p(e), -y * e / (one(T) + e))
    else
        e = exp(yp)
        (log1p(e) - yp, -y * one(T) / (one(T) + e))
    end
end

## Multinomial logistic loss (for Multinomial logistic regression)
#
#   loss(p, y) := log(sum_k exp(p[k])) - p[y]
#

type MultiLogisticLoss <: MultivariateLoss
end

function value_and_deriv!{T<:BlasReal}(::MultiLogisticLoss, p::DenseVector{T}, y::Integer)
    pmax = maximum(p)
    p_y = u[y]

    k = length(p)
    s = zero(T)
    @inbounds for i = 1:k
        pi = exp(p[i] - pmax)
        p[i] = pi
        s += pi
    end

    @inbounds for i = 1:k
        p[i] /= s
    end
    p[y] -= one(T)

    return umax - p_y + log(s)
end
