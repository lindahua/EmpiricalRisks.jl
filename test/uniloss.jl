using LossFuns
using Base.Test
using DualNumbers


### auxiliary functions

function verify_uniloss(loss::UnivariateLoss, f, u::Float64, y::Real)
    r = f(dual(u, 1.0), y)
    v_r = real(r)
    dv_r = epsilon(r)

    @test_approx_eq v_r value(loss, u, y)
    @test_approx_eq dv_r deriv(loss, u, y)

    (v, dv) = value_and_deriv(loss, u, y)
    @test_approx_eq v_r v
    @test_approx_eq dv_r dv
end

function verify_uniloss(loss::UnivariateLoss, f, us::AbstractVector, ys::AbstractVector)
    for u in us, y in ys
        verify_uniloss(loss, f, u, y)
    end
end


### test cases

# AbsLoss

_absf(u::Dual, y) = real(u) > y ? u - y :
                    real(u) < y ? y - u : dual(0.0, 0.0)

verify_uniloss(AbsLoss(), _absf, -3.0:3.0, -1.0:0.5:1.0)

# SqrLoss

verify_uniloss(SqrLoss(),
    (p, y) -> abs2(p - y) / 2, -3.0:3.0, -1.0:0.5:1.0)


# HingeLoss

_hingef(u::Dual, y) = y * real(u) < 1.0 ? 1.0 - y * u : dual(0.0, 0.0)
verify_uniloss(HingeLoss(), _hingef, -2.0:0.5:2.0, [-1.0, 1.0])

# LogisticLoss

verify_uniloss(LogisticLoss(),
    (p, y) -> log(1 + exp(-y * p)), -2.0:0.5:2.0, [-1.0, -0.5, 0.5, 1.0])
