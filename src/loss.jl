
# N is the number of dimensions of the prediction
abstract Loss{N}
typealias UnivariateLoss Loss{0}
typealias MultivariateLoss Loss{1}

## Abs loss (for regression)
#
#   loss(p, y) = |p - y|
#
immutable AbsLoss <: UnivariateLoss
end

value{T<:BlasReal}(::AbsLoss, p::T, y::T) = abs(p - y)
deriv{T<:BlasReal}(::AbsLoss, p::T, y::T) = sign(p - y)
value_and_deriv{T<:BlasReal}(::AbsLoss, p::T, y::T) = (r = p - y; (abs(r), sign(r)))


## Squared loss (for regression)
#
#   loss(p, y) := (1/2) * (p - y)^2
#
immutable SqrLoss <: UnivariateLoss
end

value{T<:BlasReal}(::SqrLoss, p::T, y::T) = half(abs2(p - y))
deriv{T<:BlasReal}(::SqrLoss, p::T, y::T) = p - y
value_and_deriv{T<:BlasReal}(::SqrLoss, p::T, y::T) = (r = p - y; v = half(abs2(r)); (v, r))


## Quantile loss (for quantile regression, asymmetric version of Abs loss)
#
#   loss(p, y) := t * (p - y)         ... (p >= y)
#               = (1 - t) * (y - p)   ... (p < y)
#
immutable QuantileLoss <: UnivariateLoss
    t::Float64

    function QuantileLoss(t::Real)
        zero(t) < t < one(t) || error("t must be a real value in (0, 1).")
        new(convert(Float64, t))
    end
end

function value{T<:BlasReal}(loss::QuantileLoss, p::T, y::T)
    t = convert(T, loss.t)
    p >= y ? t * (p - y) : (one(T) - t) * (y - p)
end

function deriv{T<:BlasReal}(loss::QuantileLoss, p::T, y::T)
    t = convert(T, loss.t)
    p > y ? t :
    p < y ? t - one(T) : zero(T)
end

function value_and_deriv{T<:BlasReal}(loss::QuantileLoss, p::T, y::T)
    t = convert(T, loss.t)
    if p > y
        (t * (p - y), t)
    elseif p < y
        c = t - one(T)
        (c * (p - y), c)
    else
        (zero(T), zero(T))
    end
end


## Epsilon Insensitive loss (for support vector regression)
#
#   loss(p, y) := 0                   ... abs(y - p) <= eps
#               = abs(y - p) - eps    ... otherwise
#
immutable EpsilonInsLoss <: UnivariateLoss
    epsilon::Float64

    function EpsilonInsLoss(epsilon::Real)
        new(convert(Float64, epsilon))
    end
end

function value{T<:BlasReal}(loss::EpsilonInsLoss, p::T, y::T)
    eps = convert(T, loss.epsilon)
    a = abs(p - y)
    a > eps ? a - eps : zero(T)
end

function deriv{T<:BlasReal}(loss::EpsilonInsLoss, p::T, y::T)
    eps = convert(T, loss.epsilon)
    r = p - y
    abs(r) > eps ? sign(r) : zero(T)
end

function value_and_deriv{T<:BlasReal}(loss::EpsilonInsLoss, p::T, y::T)
    eps = convert(T, loss.epsilon)
    r = p - y
    a = abs(r)
    a > eps ? (a - eps, sign(r)) : (zero(T), zero(T))
end


## Huber loss (for regression, smoothed version of Abs loss)
#
#   loss(p, y) := (1/2) * (p - y)^2      ... (|p - y| <= h)
#                 h * |p - y| - h^2/2    ... otherwise
#
immutable HuberLoss <: UnivariateLoss
    h::Float64

    function HuberLoss(h::Real)
        h > zero(h) || error("h must be a positive value.")
        new(convert(Float64, h))
    end
end

function value{T<:BlasReal}(loss::HuberLoss, p::T, y::T)
    h = convert(T, loss.h)
    a = abs(p - y)
    a <= h ? half(a * a) : h * a - half(h * h)
end

function deriv{T<:BlasReal}(loss::HuberLoss, p::T, y::T)
    h = convert(T, loss.h)
    r = p - y
    r > h ? h : r < -h ? -h : r
end

function value_and_deriv{T<:BlasReal}(loss::HuberLoss, p::T, y::T)
    h = convert(T, loss.h)
    r = p - y
    r > h ? (h * r - half(h * h), h) :
    r < -h ? (-h * r - half(h * h), -h) :
    (half(r * r), r)
end


## Hinge loss (for L1-SVM)
#
#   loss(p, y) := max(1 - y * p, 0)
#
immutable HingeLoss <: UnivariateLoss
end

value{T<:BlasReal}(::HingeLoss, p::T, y::T) = nonneg(one(T) - y * p)
deriv{T<:BlasReal}(::HingeLoss, p::T, y::T) = y * p < one(T) ? -y : zero(T)

function value_and_deriv{T<:BlasReal}(::HingeLoss, p::T, y::T)
    yp = y * p
    yp >= one(T) ? (zero(T), zero(T)) : (one(T) - yp, -y)
end


## Squared Hinge loss (for L2-SVM)
#
#   loss(p, y) := max(1 - y * p, 0)^2
#
immutable SqrHingeLoss <: UnivariateLoss
end

function value{T<:BlasReal}(::SqrHingeLoss, p::T, y::T)
    yp = y * p
    yp >= one(T) ? zero(T) : abs2(nonneg(one(T) - yp))
end

deriv{T<:BlasReal}(::SqrHingeLoss, p::T, y::T) = y * p < one(T) ? 2(p - y) : zero(T)

function value_and_deriv{T<:BlasReal}(::SqrHingeLoss, p::T, y::T)
    yp = y * p
    yp >= one(T) ? (zero(T), zero(T)) : (abs2(one(T) - yp), 2(p - y))
end


## Smoothed HingeLoss
#
#   loss(p, y) := 0              ... y * p > 1 + h
#                 1 - y * p      ... y * p < 1 - h
#                 (1 + h - y * p)^2 / 4h   ... otherwise
#
#  Reference
#
#   O. Chapelle, "Training a Support Vector Machine in the Primal", Neural Computation.
#
immutable SmoothedHingeLoss <: UnivariateLoss
    h::Float64

    function SmoothedHingeLoss(h::Real)
        h > zero(h) || error("h must be a positive value.")
        new(convert(Float64, h))
    end
end

function value{T<:BlasReal}(loss::SmoothedHingeLoss, p::T, y::T)
    h = convert(T, loss.h)
    yp = y * p
    yp >= one(T) + h ? zero(T) :
    yp <= one(T) - h ? one(T) - yp :
    abs2(one(T) + h - yp) / 4h
end

function deriv{T<:BlasReal}(loss::SmoothedHingeLoss, p::T, y::T)
    h = convert(T, loss.h)
    yp = y * p
    yp >= one(T) + h ? zero(T) :
    yp <= one(T) - h ? -y :
    y * (yp - one(T) - h) / 2h
end

function value_and_deriv{T<:BlasReal}(loss::SmoothedHingeLoss, p::T, y::T)
    h = convert(T, loss.h)
    yp = y * p
    if yp >= one(T) + h
        (zero(T), zero(T))
    elseif yp <= one(T) - h
        (one(T) - yp, -y)
    else
        z = yp - one(T) - h
        (abs2(z) / 4h, y * z / 2h)
    end
end


## Logistic loss (for logistic regression)
#
#   loss(p, y) := log(1 + exp(-y * p))
#
immutable LogisticLoss <: UnivariateLoss
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

grad{T<:BlasReal}(loss::MultivariateLoss, p::StridedVector{T}, y) =
    grad!(loss, zeros(T, length(p)), p, y)

value_and_grad{T<:BlasReal}(loss::MultivariateLoss, p::StridedVector{T}, y) =
    value_and_grad!(loss, zeros(T, length(p)), p, y)


## SumLoss
#
#   loss(p, y) := sum_k loss.intern(p[k], y[k])
#
immutable SumLoss{L<:UnivariateLoss} <: MultivariateLoss
    intern::L
end

SumLoss{L<:UnivariateLoss}(loss::L) = SumLoss{L}(loss)

typealias SumSqrLoss SumLoss{SqrLoss}
SumSqrLoss() = SumLoss{SqrLoss}(SqrLoss())

function value{T<:BlasReal}(s::SumLoss, p::StridedVector{T}, y::StridedVector{T})
    loss = s.intern
    k = length(p)
    @_checkdims k == length(y)
    s = value(loss, p[1], y[1])
    @inbounds for i = 2:k
        s += value(loss, p[i], y[i])
    end
    s
end

function grad!{T<:BlasReal}(s::SumLoss, g::StridedVector{T}, p::StridedVector{T}, y::StridedVector{T})
    loss = s.intern
    k = length(p)
    @_checkdims k == length(g) == length(y)
    @inbounds for i = 1:k
        g[i] = deriv(loss, p[i], y[i])
    end
    g
end

function value_and_grad!{T<:BlasReal}(s::SumLoss, g::StridedVector{T}, p::StridedVector{T}, y::StridedVector{T})
    loss = s.intern
    k = length(p)
    @_checkdims k == length(g) == length(y)
    s, dv = value_and_deriv(loss, p[1], y[1])
    g[1] = dv
    @inbounds for i = 2:k
        v, dv = value_and_deriv(loss, p[i], y[i])
        s += v
        g[i] = dv
    end
    (s, g)
end


## Multinomial logistic loss (for Multinomial logistic regression)
#
#   loss(p, y) := log(sum_k exp(p[k])) - p[y]
#
immutable MultiLogisticLoss <: MultivariateLoss
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
    @_checkdims length(g) == k
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
    @_checkdims length(g) == k

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
