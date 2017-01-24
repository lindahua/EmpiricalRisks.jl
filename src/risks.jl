# Risk model types

abstract RiskModel

immutable SupervisedRiskModel{PM<:PredictionModel,L<:Loss} <: RiskModel
    predmodel::PM
    loss::L
end

riskmodel{N,M}(pm::PredictionModel{N,M}, loss::Loss{M}) =
    SupervisedRiskModel{typeof(pm), typeof(loss)}(pm,loss)

### generic functions

function value{PM<:PredictionModel,L<:Loss}(rm::SupervisedRiskModel{PM,L}, θ, x, y)
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

function value!{PM<:PredictionModel,L<:Loss,T<:BlasReal}(buffer, rm::SupervisedRiskModel{PM,L}, θ, X::StridedMatrix{T}, y)
    pm = rm.predmodel
    loss = rm.loss
    n = ninputs(pm, X)
    n == size(y, ndims(y)) || error("Unmatched inputs and outputs")
    predict!(buffer, pm, θ, X)
    @inbounds s = value(loss, gets(buffer,1), gets(y,1))
    for i = 2:n
        @inbounds s += value(loss, gets(buffer,i), gets(y,i))
    end
    s
end

value_and_grad(rm::SupervisedRiskModel, θ, x, y) = value_and_addgrad!(rm, zero(eltype(θ)), similar(θ), one(eltype(θ)), θ, x, y)


### Univariate prediction + Univariate loss

function value_and_addgrad!{T<:BlasReal,L<:UnivariateLoss}(rm::SupervisedRiskModel{LinearPred,L},
                                                           β::Real, g::StridedVector{T},
                                                           α::Real, θ::StridedVector{T}, x::StridedVector{T}, y::Real)
    pm = rm.predmodel
    loss = rm.loss
    @_checkdims size(g) == size(θ) == paramsize(pm)
    n = ninputs(pm, x)
    @assert n == 1

    p = dot(θ, x)
    v, dv = value_and_deriv(loss, p, y)
    α_ = convert(T, α)
    axpby!(convert(T, α_ * dv), x, convert(T, β), g)
    (α_ * v, g)
end

function value_and_addgrad!{T<:BlasReal,L<:UnivariateLoss}(rm::SupervisedRiskModel{LinearPred,L},
                                                           β::Real, g::StridedVector{T},
                                                           α::Real, θ::StridedVector{T}, x::StridedMatrix{T}, y::StridedVector)
    buffer = zeros(size(x,2))
    value_and_addgrad!(buffer, rm, β, g, α, θ, x, y)
end

function value_and_addgrad!{T<:BlasReal,L<:UnivariateLoss}(buffer::StridedVecOrMat{T}, rm::SupervisedRiskModel{LinearPred,L},
                                                           β::Real, g::StridedVector{T},
                                                           α::Real, θ::StridedVector{T}, x::StridedMatrix{T}, y::StridedVector)
    pm = rm.predmodel
    loss = rm.loss
    @_checkdims size(g) == size(θ) == paramsize(pm)
    n = ninputs(pm, x)
    n == length(y) || error("Unmatched inputs and outputs.")

    u = predict!(buffer, pm, θ, x)
    @assert length(u) == n
    v = zero(T)
    @inbounds for i = 1:n
        v_i, u_i = value_and_deriv(loss, u[i], y[i])
        v += v_i
        u[i] = u_i
    end
    α_ = convert(T, α)
    gemv!('N', α_, x, u, convert(T, β), g)
    (α_ * v, g)
end

function value_and_addgrad!{T<:BlasReal,L<:UnivariateLoss}(rm::SupervisedRiskModel{AffinePred,L},
                                                           β::Real, g::StridedVector{T},
                                                           α::Real, θ::StridedVector{T}, x::StridedVector{T}, y::Real)
    pm = rm.predmodel
    loss = rm.loss
    @_checkdims size(g) == size(θ) == paramsize(pm)
    n = ninputs(pm, x)
    @assert n == 1

    d = inputlen(pm)
    p = predict(pm, θ, x)
    v, dv = value_and_deriv(loss, p, y)
    α_ = convert(T, α)
    β_ = convert(T, β)
    axpby!(α_ * dv, x, β_, Base.view(g, 1:d))
    gb = dv * convert(T, pm.bias)
    if β == zero(T)
        g[d+1] = α_ * gb
    else
        g[d+1] = β_ * g[d+1] + α_ * gb
    end
    (α_ * v, g)
end

function value_and_addgrad!{T<:BlasReal,L<:UnivariateLoss}(rm::SupervisedRiskModel{AffinePred,L},
                                                           β::Real, g::StridedVector{T},
                                                           α::Real, θ::StridedVector{T}, x::StridedMatrix{T}, y::StridedVector)
    buffer = zeros(size(x,2))
    value_and_addgrad!(buffer, rm, β, g, α, θ, x, y)
end

function value_and_addgrad!{T<:BlasReal,L<:UnivariateLoss}(buffer::StridedVecOrMat{T}, rm::SupervisedRiskModel{AffinePred,L},
                                                           β::Real, g::StridedVector{T},
                                                           α::Real, θ::StridedVector{T}, x::StridedMatrix{T}, y::StridedVector)
    pm = rm.predmodel
    loss = rm.loss
    @_checkdims size(g) == size(θ) == paramsize(pm)
    n = ninputs(pm, x)
    n == length(y) || error("Unmatched inputs and outputs.")

    d = inputlen(pm)
    u = predict!(buffer, pm, θ, x)
    @assert length(u) == n
    v = zero(T)
    @inbounds for i = 1:n
        v_i, u_i = value_and_deriv(loss, u[i], y[i])
        v += v_i
        u[i] = u_i
    end
    α_ = convert(T, α)
    β_ = convert(T, β)
    gemv!('N', α_, x, u, β_, Base.view(g, 1:d))
    gb = convert(T, sum(u) * pm.bias)
    if β == zero(T)
        g[d+1] = α_ * gb
    else
        g[d+1] = β_ * g[d+1] + α_ * gb
    end
    (α_ * v, g)
end

### Multivariate prediction + Multivariate loss

function value_and_addgrad!{T<:BlasReal,L<:MultivariateLoss}(rm::SupervisedRiskModel{MvLinearPred,L},
                                                             β::Real, g::StridedMatrix{T},
                                                             α::Real, θ::StridedMatrix{T}, x::StridedVector{T}, y)
    pm = rm.predmodel
    loss = rm.loss
    @_checkdims size(g) == size(θ) == paramsize(pm)
    n = ninputs(pm, x)
    @assert n == 1

    u = predict(pm, θ, x)
    v, _ = value_and_grad!(loss, u, u, y)
    α_ = convert(T, α)
    gemm!('N', 'T', α_, u, x, convert(T, β), g)
    (α_ * v, g)
end

function value_and_addgrad!{T<:BlasReal,L<:MultivariateLoss}(rm::SupervisedRiskModel{MvLinearPred,L},
                                                             β::Real, g::StridedMatrix{T},
                                                             α::Real, θ::StridedMatrix{T}, x::StridedMatrix{T}, y)
    buffer = zeros(size(θ,1), size(x,2))
    value_and_addgrad!(buffer, rm, β, g, α, θ, x, y)
end

function value_and_addgrad!{T<:BlasReal,L<:MultivariateLoss}(buffer::StridedVecOrMat{T}, rm::SupervisedRiskModel{MvLinearPred,L},
                                                             β::Real, g::StridedMatrix{T},
                                                             α::Real, θ::StridedMatrix{T}, x::StridedMatrix{T}, y)
    pm = rm.predmodel
    loss = rm.loss
    @_checkdims size(g) == size(θ) == paramsize(pm)
    n = ninputs(pm, x)
    n == size(y, ndims(y)) || error("Unmatched inputs and outputs.")

    u = predict!(buffer, pm, θ, x)
    @assert size(u, 2) == n
    v = zero(T)
    for i = 1:n
        u_i = Base.view(u,:,i)
        v_i, _ = value_and_grad!(loss, u_i, u_i, gets(y,i))
        v += v_i
    end
    α_ = convert(T, α)
    gemm!('N', 'T', α_, u, x, convert(T, β), g)
    (α_ * v, g)
end

function value_and_addgrad!{T<:BlasReal,L<:MultivariateLoss}(rm::SupervisedRiskModel{MvAffinePred,L},
                                                             β::Real, g::StridedMatrix{T},
                                                             α::Real, θ::StridedMatrix{T}, x::StridedVector{T}, y)
    pm = rm.predmodel
    loss = rm.loss
    @_checkdims size(g) == size(θ) == paramsize(pm)
    n = ninputs(pm, x)
    @assert n == 1

    u = predict(pm, θ, x)
    v, _ = value_and_grad!(loss, u, u, y)
    d = inputlen(pm)
    α_ = convert(T, α)
    β_ = convert(T, β)
    gemm!('N', 'T', α_, u, x, β_, Base.view(g, :, 1:d))
    axpby!(convert(T, α_ * pm.bias), u, β_, Base.view(g, :, d+1))
    (α_ * v, g)
end

function value_and_addgrad!{T<:BlasReal,L<:MultivariateLoss}(rm::SupervisedRiskModel{MvAffinePred,L},
                                                             β::Real, g::StridedMatrix{T},
                                                             α::Real, θ::StridedMatrix{T}, x::StridedMatrix{T}, y)
    buffer = zeros(size(θ,1), size(x,2))
    value_and_addgrad!(buffer, rm, β, g, α, θ, x, y)
end

function value_and_addgrad!{T<:BlasReal,L<:MultivariateLoss}(buffer::StridedVecOrMat{T}, rm::SupervisedRiskModel{MvAffinePred,L},
                                                             β::Real, g::StridedMatrix{T},
                                                             α::Real, θ::StridedMatrix{T}, x::StridedMatrix{T}, y)
    pm = rm.predmodel
    loss = rm.loss
    @_checkdims size(g) == size(θ) == paramsize(pm)
    n = ninputs(pm, x)
    n == size(y, ndims(y)) || error("Unmatched inputs and outputs.")

    u = predict!(buffer, pm, θ, x)
    @assert size(u, 2) == n
    v = zero(T)
    for i = 1:n
        u_i = Base.view(u,:,i)
        v_i, _ = value_and_grad!(loss, u_i, u_i, gets(y,i))
        v += v_i
    end
    d = inputlen(pm)
    α_ = convert(T, α)
    β_ = convert(T, β)
    gemm!('N', 'T', α_, u, x, β_, Base.view(g, :, 1:d))
    axpby!(convert(T, α_ * pm.bias), vec(sum(u,2)), β_, Base.view(g, :, d+1))
    (α_ * v, g)
end
