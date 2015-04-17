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

_abs(u::Dual) = real(u) == 0.0 ? dual(0.0, 0.0) : abs(u)

verify_uniloss(AbsLoss(),
    (p, y) -> _abs(p - y), -3.0:3.0, -1.0:0.5:1.0)

# SqrLoss

verify_uniloss(SqrLoss(),
    (p, y) -> abs2(p - y) / 2, -3.0:3.0, -1.0:0.5:1.0)

# HuberLoss

function _huberf(t::Float64, u::Dual, y)
    a = abs(real(u) - y)
    a > t ? t * _abs(u - y) - 0.5 * t^2 : 0.5 * abs2(u - y)
end

verify_uniloss(HuberLoss(0.3), (p, y) -> _huberf(0.3, p, y), -2.0:0.25:2.0, -1.0:0.5:1.0)
verify_uniloss(HuberLoss(0.5), (p, y) -> _huberf(0.5, p, y), -2.0:0.25:2.0, -1.0:0.5:1.0)


# HingeLoss

_hingef(u::Dual, y) = y * real(u) < 1.0 ? 1.0 - y * u : dual(0.0, 0.0)
verify_uniloss(HingeLoss(), _hingef, -2.0:0.5:2.0, [-1.0, 1.0])

# LogisticLoss

verify_uniloss(LogisticLoss(),
    (p, y) -> log(1 + exp(-y * p)), -2.0:0.5:2.0, [-1.0, -0.5, 0.5, 1.0])
