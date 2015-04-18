# Risk model types

abstract RiskModel

immutable SupervisedRiskModel{PM<:PredictionModel,L<:Loss} <: RiskModel
    predmodel::PM
    loss::L
end

riskmodel{PM<:PredictionModel,L<:Loss}(pm::PM, loss::L) = SupervisedRiskModel{PM,L}(pm,loss)

### generic functions

grad(rm::SupervisedRiskModel, θ, x, y) = addgrad!(rm, zero(eltype(θ)), similar(θ), one(eltype(θ)), θ, x, y)


### Univariate prediction + Univariate loss

function risk{PM<:UnivariatePredictionModel,L<:UnivariateLoss}(rm::SupervisedRiskModel{PM,L}, θ, x, y::Real)
    pm = rm.predmodel
    loss = rm.loss
    n = ninputs(pm, x)
    n == 1 || error("Unmatched inputs and outputs.")
    p = predict(pm, θ, x)
    value(loss, p[1], y)
end

function risk{PM<:UnivariatePredictionModel,L<:UnivariateLoss}(rm::SupervisedRiskModel{PM,L}, θ, x, y::AbstractVector)
    pm = rm.predmodel
    loss = rm.loss
    n = ninputs(pm, x)
    n == length(y) || error("Unmatched inputs and outputs.")
    p = predict(pm, θ, x)
    s = value(loss, p[1], y[1])
    for i = 2:n
        s += value(loss, p[i], y[i])
    end
    return s
end

function addgrad!{T<:BlasReal,L<:UnivariateLoss}(rm::SupervisedRiskModel{LinearPred,L},
                                                 β::Real, g::StridedVector{T},
                                                 α::Real, θ::StridedVector{T}, x::StridedVector{T}, y::Real)
    pm = rm.predmodel
    loss = rm.loss
    @_checkdims size(g) == size(θ) == paramsize(pm)
    n = ninputs(pm, x)
    @assert n == 1

    p = dot(θ, x)
    dv = deriv(loss, p, y)
    axpby!(convert(T, α * dv), x, convert(T, β), g)
    g
end

function addgrad!{T<:BlasReal,L<:UnivariateLoss}(rm::SupervisedRiskModel{LinearPred,L},
                                                 β::Real, g::StridedVector{T},
                                                 α::Real, θ::StridedVector{T}, x::StridedMatrix{T}, y::StridedVector)

    pm = rm.predmodel
    loss = rm.loss
    @_checkdims size(g) == size(θ) == paramsize(pm)
    n = ninputs(pm, x)
    n == length(y) || error("Unmatched inputs and outputs.")

    u = x'θ
    @assert length(u) == n
    @inbounds for i = 1:n
        u[i] = deriv(loss, u[i], y[i])
    end
    gemv!('N', convert(T, α), x, u, convert(T, β), g)
    g
end

function addgrad!{T<:BlasReal,L<:UnivariateLoss}(rm::SupervisedRiskModel{AffinePred,L},
                                                 β::Real, g::StridedVector{T},
                                                 α::Real, θ::StridedVector{T}, x::StridedVector{T}, y::Real)
    pm = rm.predmodel
    loss = rm.loss
    @_checkdims size(g) == size(θ) == paramsize(pm)
    n = ninputs(pm, x)
    @assert n == 1

    d = inputlen(pm)
    p = predict(pm, θ, x)
    dv = deriv(loss, p, y)
    α_ = convert(T, α)
    β_ = convert(T, β)
    axpby!(α_ * dv, x, β_, view(g, 1:d))
    gb = dv * convert(T, pm.bias)
    if β == zero(T)
        g[d+1] = α_ * gb
    else
        g[d+1] = β_ * g[d+1] + α_ * gb
    end
    g
end

function addgrad!{T<:BlasReal,L<:UnivariateLoss}(rm::SupervisedRiskModel{AffinePred,L},
                                                 β::Real, g::StridedVector{T},
                                                 α::Real, θ::StridedVector{T}, x::StridedMatrix{T}, y::StridedVector)
    pm = rm.predmodel
    loss = rm.loss
    @_checkdims size(g) == size(θ) == paramsize(pm)
    n = ninputs(pm, x)
    n == length(y) || error("Unmatched inputs and outputs.")

    d = inputlen(pm)
    u = predict(pm, θ, x)
    @assert length(u) == n
    @inbounds for i = 1:n
        u[i] = deriv(loss, u[i], y[i])
    end
    α_ = convert(T, α)
    β_ = convert(T, β)
    gemv!('N', α_, x, u, β_, view(g, 1:d))
    gb = convert(T, sum(u) * pm.bias)
    if β == zero(T)
        g[d+1] = α_ * gb
    else
        g[d+1] = β_ * g[d+1] + α_ * gb
    end
    g
end
