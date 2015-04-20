using EmpiricalRisks
using Base.Test
import EmpiricalRisks: gets

## auxiliary testing facilities

function _addgrad(pm::PredictionModel, loss::Loss,
                  β::Float64, g0::StridedVecOrMat, α::Float64, θ::StridedVecOrMat, X, y)

    rm = riskmodel(pm, loss)
    addgrad!(rm, β, copy(g0), α, θ, X, y)
end


function verify_risk_values(pm::PredictionModel, loss::Loss,
                      θ::VecOrMat{Float64}, X::Matrix{Float64}, y::VecOrMat{Float64}, rr::Vector{Float64})

    n = size(X, 2)
    rm = riskmodel(pm, loss)

    # over individual samples
    for i = 1:n
        x_i = gets(X, i)
        y_i = gets(y, i)
        @test_approx_eq rr[i] risk(rm, θ, x_i, y_i)
    end

    # over sample batch
    @test_approx_eq sum(rr) risk(rm, θ, X, y)
end

function verify_risk_grads(pm::PredictionModel, loss::Loss,
                           θ::VecOrMat{Float64}, X::Matrix{Float64}, y::VecOrMat{Float64}, G::Array{Float64})

    n = size(X, 2)
    rm = riskmodel(pm, loss)
    g0 = randn(size(θ))

    # over individual samples
    for i = 1:n
        x_i = gets(X, i)
        y_i = gets(y, i)
        g_i = gets(G, i)
        @test_approx_eq g_i grad(rm, θ, x_i, y_i)
        @test_approx_eq g0 + g_i _addgrad(pm, loss, 1.0, g0, 1.0, θ, x_i, y_i)
        @test_approx_eq g0 + 2.0 * g_i _addgrad(pm, loss, 1.0, g0, 2.0, θ, x_i, y_i)
        @test_approx_eq 0.4 * g0 + 0.8 * g_i _addgrad(pm, loss, 0.4, g0, 0.8, θ, x_i, y_i)
    end

    # over sample batch
    g = gets(sum(G, ndims(G)), 1)
    @test_approx_eq g grad(rm, θ, X, y)
    @test_approx_eq g0 + g _addgrad(pm, loss, 1.0, g0, 1.0, θ, X, y)
    @test_approx_eq g0 + 2.0 * g _addgrad(pm, loss, 1.0, g0, 2.0, θ, X, y)
    @test_approx_eq 0.4 * g0 + 0.8 * g _addgrad(pm, loss, 0.4, g0, 0.8, θ, X, y)
end


function verify_risk(pm::PredictionModel, loss::Loss,
                     θ::VecOrMat{Float64}, X::Matrix{Float64}, y::VecOrMat{Float64})


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
    verify_risk_values(pm, loss, θ, X, y, R0)
    verify_risk_grads(pm, loss, θ, X, y, G0)
end


## mock a multi-valued sqr loss for testing

immutable MSqrLoss <: MultivariateLoss
end

EmpiricalRisks.value(::MSqrLoss, u::StridedVector, y::StridedVector) = sumabs2(u - y) / 2

function EmpiricalRisks.grad!(::MSqrLoss, g::StridedVector, u::StridedVector, y::StridedVector)
    for i = 1:length(g)
        g[i] = u[i] - y[i]
    end
    return g
end

## (safe) reference implementation to compare with

function _risk_and_grad(::LinearPred, ::SqrLoss, w::StridedVector, x::StridedVector, y::Real)
    r = dot(w, x) - y
    (r^2 / 2, x * r)
end

function _risk_and_grad(pm::AffinePred, ::SqrLoss, wa::StridedVector, x::StridedVector, y::Real)
    d = length(x)
    w, a = wa[1:d], wa[d+1]
    p = dot(w, x) + a * pm.bias
    r = p - y
    (r^2 / 2, [x; pm.bias] * r)
end

function _risk_and_grad(::MvLinearPred, ::MSqrLoss, W::StridedMatrix, x::StridedVector, y::StridedVector)
    r = W * x - y
    (sumabs2(r) / 2, r * x')
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

verify_risk(MvLinearPred(d, k), MSqrLoss(), W, X, y)
