# Risk model types

abstract RiskModel

immutable SupervisedRiskModel{PM<:PredictionModel,L<:Loss} <: RiskModel
    predmodel::PM
    loss::L
end

riskmodel{N,M}(pm::PredictionModel{N,M}, loss::Loss{M}) =
    SupervisedRiskModel{typeof(pm), typeof(loss)}(pm,loss)

### generic functions

grad(rm::SupervisedRiskModel, θ, x, y) = addgrad!(rm, zero(eltype(θ)), similar(θ), one(eltype(θ)), θ, x, y)

function risk{PM<:PredictionModel,L<:Loss}(rm::SupervisedRiskModel{PM,L}, θ, x, y)
    pm = rm.predmodel
    loss = rm.loss
    if size(x) == inputsize(pm)
        p = predict(pm, θ, x)
        value(loss, p, y)
    else
        n = ninputs(pm, x)
        n == size(y, ndims(y)) || error("Unmatched inputs and outputs")
        p = predict(pm, θ, x)
        s = value(loss, gets(p,1), gets(y,1))
        for i = 2:n
            s += value(loss, gets(p,i), gets(y,i))
        end
        s
    end
end


### Univariate prediction + Univariate loss

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


### Multivariate prediction + Multivariate loss

function addgrad!{T<:BlasReal,L<:MultivariateLoss}(rm::SupervisedRiskModel{MvLinearPred,L},
                                                   β::Real, g::StridedMatrix{T},
                                                   α::Real, θ::StridedMatrix{T}, x::StridedVector{T}, y)
    pm = rm.predmodel
    loss = rm.loss
    @_checkdims size(g) == size(θ) == paramsize(pm)
    n = ninputs(pm, x)
    @assert n == 1

    u = predict(pm, θ, x)
    grad!(loss, u, u, y)
    gemm!('N', 'T', convert(T, α), u, x, convert(T, β), g)
    g
end

function addgrad!{T<:BlasReal,L<:MultivariateLoss}(rm::SupervisedRiskModel{MvLinearPred,L},
                                                   β::Real, g::StridedMatrix{T},
                                                   α::Real, θ::StridedMatrix{T}, x::StridedMatrix{T}, y)

    pm = rm.predmodel
    loss = rm.loss
    @_checkdims size(g) == size(θ) == paramsize(pm)
    n = ninputs(pm, x)
    n == size(y, ndims(y)) || error("Unmatched inputs and outputs.")

    u = predict(pm, θ, x)
    @assert size(u, 2) == n
    for i = 1:n
        u_i = view(u,:,i)
        grad!(loss, u_i, u_i, gets(y,i))
    end
    gemm!('N', 'T', convert(T, α), u, x, convert(T, β), g)
    g
end
