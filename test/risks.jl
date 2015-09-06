using EmpiricalRisks
using Base.Test
import EmpiricalRisks: gets

## auxiliary testing facilities

function _val_and_addgrad(rm::SupervisedRiskModel,
                          β::Float64, g0::StridedVecOrMat, α::Float64, θ::StridedVecOrMat, X, y)
    value_and_addgrad!(rm, β, copy(g0), α, θ, X, y)
end

function verify_risk(pm::PredictionModel, loss::Loss,
                     θ::VecOrMat{Float64}, X::Matrix{Float64}, y::VecOrMat)

    rm = riskmodel(pm, loss)

    # produce ground-truth
    n = size(X, 2)
    @assert ninputs(pm, X) == n
    @assert size(y, ndims(y)) == n

    R0 = zeros(n)
    if ndims(θ) == 1
        d = length(θ)
        G0 = zeros(d, n)
        for i = 1:n
            r, g = _risk_and_grad(pm, loss, θ, gets(X,i), gets(y,i))
            R0[i] = r
            G0[:,i] = g
        end
    else
        d1, d2 = size(θ)
        G0 = zeros(d1, d2, n)
        for i = 1:n
            r, g = _risk_and_grad(pm, loss, θ, gets(X,i), gets(y,i))
            R0[i] = r
            G0[:,:,i] = g
        end
    end

    # perform verification

    # over individual samples
    g0 = rand(size(θ))

    for i = 1:n
        x_i = gets(X, i)
        y_i = gets(y, i)
        g_i = gets(G0, i)
        v_i = R0[i]

        @test_approx_eq v_i value(rm, θ, x_i, y_i)

        v, g = value_and_grad(rm, θ, x_i, y_i)
        @test_approx_eq v_i v
        @test_approx_eq g_i g

        for β in [0.0, 0.5, 1.0], α in [0.5, 1.0, 2.0]
            v, g = _val_and_addgrad(rm, β, g0, α, θ, x_i, y_i)
            @test_approx_eq α * v_i v
            @test_approx_eq β * g0 + α * g_i g
        end
    end

    # over sample batch
    rv = sum(R0)
    rg = gets(sum(G0, ndims(G0)), 1)

    @test_approx_eq rv value(rm, θ, X,y)

    buffer = ndims(θ) == 2 ? zeros(size(θ,1), size(X,2)) : zeros(size(X,2))
    @test_approx_eq rv value!(buffer, rm, θ, X, y)

    v, g = value_and_grad(rm, θ, X, y)
    @test_approx_eq rv v
    @test_approx_eq rg g

    for β in [0.0, 0.5, 1.0], α in [0.5, 1.0, 2.0]
        v, g = _val_and_addgrad(rm, β, g0, α, θ, X, y)
        @test_approx_eq α * rv v
        @test_approx_eq β * g0 + α * rg g
    end
end



## (safe) reference implementation to compare with

function _risk_and_grad(::LinearPred, ::SqrLoss, w::StridedVector, x::StridedVector, y::Real)
    r = dot(w, x) - y
    (r^2 / 2, x * r)
end

function _risk_and_grad(pm::AffinePred, ::SqrLoss, wa::StridedVector, x::StridedVector, y::Real)
    x_ = [x; pm.bias]
    r = dot(wa, x_) - y
    (r^2 / 2, x_ * r)
end

function _risk_and_grad(::MvLinearPred, ::SumSqrLoss, W::StridedMatrix, x::StridedVector, y::StridedVector)
    r = W * x - y
    (sumabs2(r) / 2, r * x')
end

function _risk_and_grad(pm::MvAffinePred, ::SumSqrLoss, Wa::StridedMatrix, x::StridedVector, y::StridedVector)
    d = length(x)
    x_ = [x; pm.bias]
    p = Wa * x_
    r = p - y
    (sumabs2(r) / 2, r * x_')
end

function _risk_and_grad(pm::MvLinearPred, ::MultiLogisticLoss, W::StridedMatrix, x::StridedVector, y::Int)
    p = W * x
    ep = exp(p)
    q = ep ./ sum(ep)
    q[y] -= 1
    (log(sum(ep)) - p[y], q * x')
end


### Univariate prediction + Univariate Loss

d = 5
n = 8
a = randn()
bias = 2.5
w = randn(d)
wa = [w; a]
X = randn(d, n)
y = randn(n)

verify_risk(LinearPred(d), SqrLoss(), w, X, y)
verify_risk(AffinePred(d), SqrLoss(), wa, X, y)
verify_risk(AffinePred(d, bias), SqrLoss(), wa, X, y)

### Multivariate prediction + Univariate Loss

k = 3
W = randn(k, d)
a = randn(k)
Wa = [W a]
X = randn(d, n)
y = randn(k, n)

verify_risk(MvLinearPred(d, k), SumSqrLoss(), W, X, y)
verify_risk(MvAffinePred(d, k), SumSqrLoss(), Wa, X, y)
verify_risk(MvAffinePred(d, k, bias), SumSqrLoss(), Wa, X, y)

y = [1, 2, 3, 3, 2, 1, 2, 3]
@assert length(y) == n
verify_risk(MvLinearPred(d, k), MultiLogisticLoss(), W, X, y)
