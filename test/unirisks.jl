using EmpiricalRisks
using Base.Test


function _addgrad(pm::UnivariatePredictionModel, loss::UnivariateLoss,
                  β::Float64, g0::Vector{Float64}, α::Float64, θ::Vector{Float64}, X, y)

    rm = riskmodel(pm, loss)
    addgrad!(rm, β, copy(g0), α, θ, X, y)
end


function verify_risks(pm::UnivariatePredictionModel, loss::UnivariateLoss,
                      θ::Vector{Float64}, X::Matrix{Float64}, y::Vector, rr::Vector{Float64})

    n = size(X, 2)
    rm = riskmodel(pm, loss)

    # over individual samples
    for i = 1:n
        x_i = X[:,i]
        @test_approx_eq rr[i] risk(rm, θ, x_i, y[i])
    end

    # over sample batch
    @test_approx_eq sum(rr) risk(rm, θ, X, y)
end

function verify_riskgrad(pm::UnivariatePredictionModel, loss::UnivariateLoss,
                         θ::Vector{Float64}, X::Matrix{Float64}, y::Vector, G::Matrix{Float64})

    n = size(X, 2)
    rm = riskmodel(pm, loss)
    g0 = randn(size(θ))

    # over individual samples
    for i = 1:n
        x_i = X[:,i]
        g_i = G[:,i]
        @test_approx_eq g_i grad(rm, θ, x_i, y[i])
        @test_approx_eq g0 + g_i _addgrad(pm, loss, 1.0, g0, 1.0, θ, x_i, y[i])
        @test_approx_eq g0 + 2.0 * g_i _addgrad(pm, loss, 1.0, g0, 2.0, θ, x_i, y[i])
        @test_approx_eq 0.4 * g0 + 0.8 * g_i _addgrad(pm, loss, 0.4, g0, 0.8, θ, x_i, y[i])
    end

    # over sample batch
    g = G * ones(n)
    @test_approx_eq g grad(rm, θ, X, y)
    @test_approx_eq g0 + g _addgrad(pm, loss, 1.0, g0, 1.0, θ, X, y)
    @test_approx_eq g0 + 2.0 * g _addgrad(pm, loss, 1.0, g0, 2.0, θ, X, y)
    @test_approx_eq 0.4 * g0 + 0.8 * g _addgrad(pm, loss, 0.4, g0, 0.8, θ, X, y)
end


## Data

d = 5
n = 8
a = randn()
bias = 2.5
w = randn(d)
wa = [w; a]
X = randn(d, n)
y = randn(n)

## LineaPred

loss = SqrLoss()

pm = LinearPred(d)

r = X'w - y
R0 = abs2(r) * 0.5
G0 = X .* (r')

verify_risks(pm, loss, w, X, y, R0)
verify_riskgrad(pm, loss, w, X, y, G0)

## AffinePred

pm = AffinePred(d, bias)

r = (X'w .+ a*bias) - y
R0 = abs2(r) * 0.5
G0 = [X; bias * ones(1, n)] .* (r')

verify_risks(pm, loss, wa, X, y, R0)
verify_riskgrad(pm, loss, wa, X, y, G0)
