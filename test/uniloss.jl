using EmpiricalRisks
using Base.Test
using DualNumbers
using Compat

### auxiliary functions

function verify_uniloss(loss::UnivariateLoss, f, u::Float64, y::Real)
    # verify inferred types
    for VT in [Float64, Float32]
        @test Base.return_types(value, @compat Tuple{typeof(loss), VT, VT}) == [VT]
        @test Base.return_types(deriv, @compat Tuple{typeof(loss), VT, VT}) == [VT]
        @test Base.return_types(value_and_deriv, @compat Tuple{typeof(loss), VT, VT}) == [@compat Tuple{VT, VT}]
    end

    # verify computation correctness
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

# QuantileLoss

function _quanlossf(t::Float64, u::Dual, y)
    real(u) > y ? t * (u - y) :
    real(u) < y ? (1.0 - t) * (y - u) :
    dual(0.0, 0.0)
end

verify_uniloss(QuantileLoss(0.3), (p, y) -> _quanlossf(0.3, p, y), -2.0:0.5:2.0, -1.0:0.5:1.0)
verify_uniloss(QuantileLoss(0.5), (p, y) -> _quanlossf(0.5, p, y), -2.0:0.5:2.0, -1.0:0.5:1.0)

# HuberLoss

function _huberf(h::Float64, u::Dual, y)
    a = abs(real(u) - y)
    a > h ? h * _abs(u - y) - 0.5 * h^2 : 0.5 * abs2(u - y)
end

verify_uniloss(HuberLoss(0.3), (p, y) -> _huberf(0.3, p, y), -2.0:0.25:2.0, -1.0:0.5:1.0)
verify_uniloss(HuberLoss(0.5), (p, y) -> _huberf(0.5, p, y), -2.0:0.25:2.0, -1.0:0.5:1.0)


# HingeLoss

_hingef(u::Dual, y) = y * real(u) < 1.0 ? 1.0 - y * u : dual(0.0, 0.0)
verify_uniloss(HingeLoss(), _hingef, -2.0:0.5:2.0, [-1.0, 1.0])


# SquaredHingeLoss

_sqrhingef(u::Dual, y) = y * real(u) < 1.0 ? (1.0 - y * u).^2 : dual(0.0, 0.0)
verify_uniloss(SqrHingeLoss(), _sqrhingef, -2.0:0.5:2.0, [-1.0, 1.0])


# SmoothedHingeLoss

function _sm_hingef(h::Float64, u::Dual, y)
    yu = y * real(u)
    yu >= 1.0 + h ? dual(0.0, 0.0) :
    yu <= 1.0 - h ? 1.0 - y * u :
    abs2(1.0 + h - y * u) / (4 * h)
end

verify_uniloss(SmoothedHingeLoss(0.2), (p, y) -> _sm_hingef(0.2, p, y), -2.0:0.25:2.0, -1.0:0.5:1.0)
verify_uniloss(SmoothedHingeLoss(0.5), (p, y) -> _sm_hingef(0.5, p, y), -2.0:0.25:2.0, -1.0:0.5:1.0)

# LogisticLoss

verify_uniloss(LogisticLoss(),
    (p, y) -> log(1 + exp(-y * p)), -2.0:0.5:2.0, [-1.0, -0.5, 0.5, 1.0])
