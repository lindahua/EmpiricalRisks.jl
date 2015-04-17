using LossFuns
using Base.Test
using DualNumbers

### Auxiliary functions

function verify_multiloss(loss::MultivariateLoss, f, u::Vector{Float64}, y)
    # verify inferred types
    YT = typeof(y)
    for VT in [Float64, Float32]
        @test Base.return_types(value, (typeof(loss), Vector{VT}, YT)) == [VT]
        @test Base.return_types(grad, (typeof(loss), Vector{VT}, YT)) == [Vector{VT}]
        @test Base.return_types(value_and_grad, (typeof(loss), Vector{VT}, YT)) == [(VT, Vector{VT})]
    end

    # verify computation correctness

    # prepare ground-truth
    d = length(u)
    v_r = f(u, y)
    g_r = zeros(d)
    for i = 1:d
        _ep = zeros(d)
        _ep[i] = 1.0
        _in = dual(u, _ep)
        _out = f(_in, y)
        @assert isa(_out, Dual{Float64})
        @assert isapprox(v_r, real(_out))
        g_r[i] = epsilon(_out)
    end

    # verify
    @test_approx_eq v_r value(loss, u, y)
    @test_approx_eq g_r grad(loss, u, y)

    (v, g) = value_and_grad(loss, u, y)
    @test_approx_eq v_r v
    @test_approx_eq g_r g
end


### Test cases

_mlogisticf(u, y::Int) = log(sum(exp(u))) - u[y]

k = 3
n = 8
u = randn(k, n)
for i = 1:n, y = 1:k
    verify_multiloss(MultiLogisticLoss(), _mlogisticf, copy(u[:,i]), y)
end
