using EmpiricalRisks
using Base.Test
import DualNumbers

### auxiliary functions

function verify_uniloss(loss::UnivariateLoss, f, u::Float64, y::Real)
    # verify inferred types
    for VT in [Float64, Float32]
        @test Base.return_types(EmpiricalRisks.value, Tuple{typeof(loss), VT, VT}) == [VT]
        @test Base.return_types(EmpiricalRisks.deriv, Tuple{typeof(loss), VT, VT}) == [VT]
        @test Base.return_types(EmpiricalRisks.value_and_deriv, Tuple{typeof(loss), VT, VT}) == [Tuple{VT, VT}]
    end

    # verify computation correctness
    r = f(DualNumbers.dual(u, 1.0), y)
    v_r = DualNumbers.realpart(r)
    dv_r = DualNumbers.epsilon(r)

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

_abs(u::DualNumbers.Dual) = DualNumbers.realpart(u) == 0.0 ? DualNumbers.dual(0.0, 0.0) : DualNumbers.abs(u)

verify_uniloss(AbsLoss(),
    (p, y) -> _abs(p - y), -3.0:3.0, -1.0:0.5:1.0)

# SqrLoss

verify_uniloss(SqrLoss(),
    (p, y) -> abs2(p - y) / 2, -3.0:3.0, -1.0:0.5:1.0)

# QuantileLoss

function _quanlossf(t::Float64, u::DualNumbers.Dual, y)
    DualNumbers.realpart(u) > y ? t * (u - y) :
    DualNumbers.realpart(u) < y ? (1.0 - t) * (y - u) :
    DualNumbers.dual(0.0, 0.0)
end

verify_uniloss(QuantileLoss(0.3), (p, y) -> _quanlossf(0.3, p, y), -2.0:0.5:2.0, -1.0:0.5:1.0)
verify_uniloss(QuantileLoss(0.5), (p, y) -> _quanlossf(0.5, p, y), -2.0:0.5:2.0, -1.0:0.5:1.0)

# EpsilonInsLoss

function _epsinsensf(eps::Float64, u::DualNumbers.Dual, y)
    a = abs(DualNumbers.realpart(u) - y)
    a > eps ? _abs(u - y) - eps : DualNumbers.dual(0.0, 0.0)
end

verify_uniloss(EpsilonInsLoss(0.3), (p, y) -> _epsinsensf(0.3, p, y), -2.0:0.25:2.0, -1.0:0.5:1.0)
verify_uniloss(EpsilonInsLoss(0.5), (p, y) -> _epsinsensf(0.5, p, y), -2.0:0.25:2.0, -1.0:0.5:1.0)

# HuberLoss

function _huberf(h::Float64, u::DualNumbers.Dual, y)
    a = abs(DualNumbers.realpart(u) - y)
    a > h ? h * _abs(u - y) - 0.5 * h^2 : 0.5 * abs2(u - y)
end

verify_uniloss(HuberLoss(0.3), (p, y) -> _huberf(0.3, p, y), -2.0:0.25:2.0, -1.0:0.5:1.0)
verify_uniloss(HuberLoss(0.5), (p, y) -> _huberf(0.5, p, y), -2.0:0.25:2.0, -1.0:0.5:1.0)

# HingeLoss

_hingef(u::DualNumbers.Dual, y) = y * DualNumbers.realpart(u) < 1.0 ? 1.0 - y * u : DualNumbers.dual(0.0, 0.0)
verify_uniloss(HingeLoss(), _hingef, -2.0:0.5:2.0, [-1.0, 1.0])

# SquaredHingeLoss

_sqrhingef(u::DualNumbers.Dual, y) = y * DualNumbers.realpart(u) < 1.0 ? (1.0 - y * u).^2 : DualNumbers.dual(0.0, 0.0)
verify_uniloss(SqrHingeLoss(), _sqrhingef, -2.0:0.5:2.0, [-1.0, 1.0])

# SmoothedHingeLoss

function _sm_hingef(h::Float64, u::DualNumbers.Dual, y)
    yu = y * DualNumbers.realpart(u)
    yu >= 1.0 + h ? DualNumbers.dual(0.0, 0.0) :
    yu <= 1.0 - h ? 1.0 - y * u :
    abs2(1.0 + h - y * u) / (4 * h)
end

verify_uniloss(SmoothedHingeLoss(0.2), (p, y) -> _sm_hingef(0.2, p, y), -2.0:0.25:2.0, -1.0:0.5:1.0)
verify_uniloss(SmoothedHingeLoss(0.5), (p, y) -> _sm_hingef(0.5, p, y), -2.0:0.25:2.0, -1.0:0.5:1.0)

# SqrSmoothedHingeLoss

function _sqrsm_hingef(g::Float64, u::Dual, y)
    yu = y * real(u)
    yu >= 1.0 - g ? (yu < 1.0 ? 0.5 / g * abs2(max(1.0 - y * u, dual(0.0, 0.0))) : dual(0.0, 0.0)) : 1.0 - g / 2.0 - y * u
end

verify_uniloss(SqrSmoothedHingeLoss(0.2), (p, y) -> _sqrsm_hingef(0.2, p, y), -2.0:0.5:2.0, [-1.0, 1.0])
verify_uniloss(SqrSmoothedHingeLoss(2), (p, y) -> _sqrsm_hingef(2., p, y), -2.0:0.5:2.0, [-1.0, 1.0])

# ModifiedHuberLoss

function _mod_huberf(u::Dual, y)
    yu = y * real(u)
    yu >= -1.0 ? abs2(max(1.0 - y * u, dual(0.0, 0.0))) : -4 * y * u
end

verify_uniloss(ModifiedHuberLoss(), (p, y) -> _mod_huberf(p, y), -2.0:0.5:2.0, [-1.0, 1.0])

# LogisticLoss

verify_uniloss(LogisticLoss(),
    (p, y) -> log(1 + exp(-y * p)), -2.0:0.5:2.0, [-1.0, -0.5, 0.5, 1.0])
