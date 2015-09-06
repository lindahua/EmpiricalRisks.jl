using EmpiricalRisks
using Base.Test


## Auxiliary functions

function verify_multipred(pred::PredictionModel{1,0}, θ, X::DenseMatrix)
    n = size(X, 2)
    @test ninputs(pred, X) == n
    rr = zeros(n)
    for i = 1:n
        rr[i] = predict(pred, θ, X[:,i])
    end
    @test_approx_eq predict(pred, θ, X) rr

    buffer = zeros(n)
    @test_approx_eq predict!(buffer, pred, θ, X) rr
end

function verify_multipred(pred::PredictionModel{1,1}, θ, X::DenseMatrix)
    n = size(X, 2)
    @test ninputs(pred, X) == n
    p = length(predict(pred, θ, X[:,1]))
    rr = zeros(p, n)
    for i = 1:n
        rr[:,i] = predict(pred, θ, X[:,i])
    end
    @test_approx_eq predict(pred, θ, X) rr

    buffer = zeros(p, n)
    @test_approx_eq predict!(buffer, pred, θ, X) rr
end


## Data preparation

d = 5
k = 3
n = 12

X = randn(d, n)

## Predictors

# LinearPred

a = randn()
w = randn(d)
wa = [w; a]
b = 2.5

pred = LinearPred(d)
@test inputlen(pred) == d
@test inputsize(pred) == (d,)
@test outputlen(pred) == 1
@test outputsize(pred) == ()
@test paramlen(pred) == d
@test paramsize(pred) == (d,)
@test isvalidparam(pred, w)

for i = 1:n
    x_i = X[:,i]
    @test_approx_eq predict(pred, w, x_i) dot(w, x_i)
end
verify_multipred(pred, w, X)

# AffinePred

pred = AffinePred(d, b)
@test inputlen(pred) == d
@test inputsize(pred) == (d,)
@test outputlen(pred) == 1
@test outputsize(pred) == ()
@test paramlen(pred) == d + 1
@test paramsize(pred) == (d+1,)
@test isvalidparam(pred, wa)

for i = 1:n
    x_i = X[:,i]
    @test_approx_eq predict(pred, wa, x_i) dot(w, x_i) + a * b
end
verify_multipred(pred, wa, X)

# MvLinearPred

a = randn(k)
W = randn(k, d)
Wa = [W a]
b = 2.5

pred = MvLinearPred(d, k)
@test inputlen(pred) == d
@test inputsize(pred) == (d,)
@test outputlen(pred) == k
@test outputsize(pred) == (k,)
@test paramlen(pred) == d * k
@test paramsize(pred) == (k, d)
@test isvalidparam(pred, W)

for i = 1:n
    x_i = X[:,i]
    @test_approx_eq predict(pred, W, x_i) W * x_i
end
verify_multipred(pred, W, X)

# MvAffinePred

pred = MvAffinePred(d, k, b)
@test inputlen(pred) == d
@test inputsize(pred) == (d,)
@test outputlen(pred) == k
@test outputsize(pred) == (k,)
@test paramlen(pred) == (d+1) * k
@test paramsize(pred) == (k, d+1)
@test isvalidparam(pred, Wa)

for i = 1:n
    x_i = X[:,i]
    @test_approx_eq predict(pred, Wa, x_i)  W * x_i + a * b
end
verify_multipred(pred, Wa, X)
