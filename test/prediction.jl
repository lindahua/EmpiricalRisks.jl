using EmpiricalRisks
using Base.Test


## Auxiliary functions

function verify_multipred(pred::UnivariatePredictionModel, θ, X::DenseMatrix)
    n = size(X, 2)
    @test nsamples(pred, X) == n
    rr = zeros(n)
    for i = 1:n
        rr[i] = predict(pred, θ, X[:,i])
    end
    @test_approx_eq predict(pred, θ, X) rr
end

function verify_multipred(pred::MultivariatePredictionModel, θ, X::DenseMatrix)
    n = size(X, 2)
    @test nsamples(pred, X) == n
    p = length(predict(pred, θ, X[:,1]))
    rr = zeros(p, n)
    for i = 1:n
        rr[:,i] = predict(pred, θ, X[:,i])
    end
    @test_approx_eq predict(pred, θ, X) rr
end


## Data preparation

d = 5
k = 3
n = 12

X = randn(d, n)

## Predictors

# LinearPred

a = randn()
θ = randn(d)
θa = [θ; a]
b = 2.5

for i = 1:n
    x_i = X[:,i]
    @test_approx_eq predict(LinearPred(), θ, x_i) dot(θ, x_i)
end
verify_multipred(LinearPred(), θ, X)

# AffinePred

for i = 1:n
    x_i = X[:,i]
    @test_approx_eq predict(AffinePred(b), θa, x_i) dot(θ, x_i) + a * b
end
verify_multipred(AffinePred(b), θa, X)

# MvLinearPred

a = randn(k)
θ = randn(d, k)
θa = [θ; a']
b = 2.5

for i = 1:n
    x_i = X[:,i]
    @test_approx_eq predict(MvLinearPred(), θ, x_i) θ'x_i
end
verify_multipred(MvLinearPred(), θ, X)

# MvAffinePred

for i = 1:n
    x_i = X[:,i]
    @test_approx_eq predict(MvAffinePred(b), θa, x_i) θ'x_i + a * b
end
verify_multipred(MvAffinePred(b), θa, X)
