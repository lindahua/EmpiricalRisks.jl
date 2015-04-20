using EmpiricalRisks
using Base.Test

import EmpiricalRisks: gets


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

## auxiliary tools

function _addgrad(pm::PredictionModel{1,1}, loss::MultivariateLoss,
                  β::Float64, g0::Matrix{Float64}, α::Float64, θ::Matrix{Float64}, X, y)

    rm = riskmodel(pm, loss)
    addgrad!(rm, β, copy(g0), α, θ, X, y)
end

function verify_risks(pm::PredictionModel{1,1}, loss::MultivariateLoss,
                      θ::Matrix{Float64}, X::Matrix{Float64}, y, rr::Vector{Float64})

    n = size(X, 2)
    rm = riskmodel(pm, loss)

    # over individual samples
    for i = 1:n
        x_i = X[:,i]
        y_i = gets(y, i)
        @test_approx_eq rr[i] risk(rm, θ, x_i, y_i)
    end

    # over sample batch
    @test_approx_eq sum(rr) risk(rm, θ, X, y)
end

function verify_riskgrad(pm::PredictionModel{1,1}, loss::MultivariateLoss,
                         θ::Vector{Float64}, X::Matrix{Float64}, y, G::Matrix{Float64})

    n = size(X, 2)
    rm = riskmodel(pm, loss)
    g0 = randn(size(θ))

    # over individual samples
    for i = 1:n
        x_i = X[:,i]
        g_i = G[:,i]
        y_i = gets(y,i)
        @test_approx_eq g_i grad(rm, θ, x_i, y_i)
        @test_approx_eq g0 + g_i _addgrad(pm, loss, 1.0, g0, 1.0, θ, x_i, y_i)
        @test_approx_eq g0 + 2.0 * g_i _addgrad(pm, loss, 1.0, g0, 2.0, θ, x_i, y_i)
        @test_approx_eq 0.4 * g0 + 0.8 * g_i _addgrad(pm, loss, 0.4, g0, 0.8, θ, x_i, y_i)
    end

    # over sample batch
    g = G * ones(n)
    @test_approx_eq g grad(rm, θ, X, y)
    @test_approx_eq g0 + g _addgrad(pm, loss, 1.0, g0, 1.0, θ, X, y)
    @test_approx_eq g0 + 2.0 * g _addgrad(pm, loss, 1.0, g0, 2.0, θ, X, y)
    @test_approx_eq 0.4 * g0 + 0.8 * g _addgrad(pm, loss, 0.4, g0, 0.8, θ, X, y)
end

function verify_riskgrad(pm::PredictionModel{1,1}, loss::MultivariateLoss,
                         θ::Matrix{Float64}, X::Matrix{Float64}, y, G::Array{Float64,3})

    n = size(X, 2)
    rm = riskmodel(pm, loss)
    g0 = randn(size(θ))

    # over individual samples
    for i = 1:n
        x_i = X[:,i]
        g_i = G[:,:,i]
        y_i = gets(y,i)
        @test_approx_eq g_i grad(rm, θ, x_i, y_i)
        @test_approx_eq g0 + g_i _addgrad(pm, loss, 1.0, g0, 1.0, θ, x_i, y_i)
        @test_approx_eq g0 + 2.0 * g_i _addgrad(pm, loss, 1.0, g0, 2.0, θ, x_i, y_i)
        @test_approx_eq 0.4 * g0 + 0.8 * g_i _addgrad(pm, loss, 0.4, g0, 0.8, θ, x_i, y_i)
    end

    # over sample batch
    g = reshape(sum(G, 3), size(G)[1:2])
    @test_approx_eq g grad(rm, θ, X, y)
    @test_approx_eq g0 + g _addgrad(pm, loss, 1.0, g0, 1.0, θ, X, y)
    @test_approx_eq g0 + 2.0 * g _addgrad(pm, loss, 1.0, g0, 2.0, θ, X, y)
    @test_approx_eq 0.4 * g0 + 0.8 * g _addgrad(pm, loss, 0.4, g0, 0.8, θ, X, y)
end


## Data

d = 5
n = 8
k = 3
a = randn(k)
bias = 2.5
W = randn(k, d)
Wa = [W a]
X = randn(d, n)
y = randn(k, n)

## LineaPred

loss = MSqrLoss()

pm = MvLinearPred(d, k)

r = W * X - y
@assert size(r) == (k, n)
R0 = vec(sum(abs2(r),1)) * 0.5
G0 = zeros(k, d, n)
for i = 1:n
    G0[:,:,i] = (W * X[:,i] - y[:,i]) * X[:,i]'
end

verify_risks(pm, loss, W, X, y, R0)
verify_riskgrad(pm, loss, W, X, y, G0)
