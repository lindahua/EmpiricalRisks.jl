using EmpiricalRisks
using Base.Test

## auxiliary functions

function _addgrad(f::Regularizer, β::Real, g0::StridedArray, α::Real, θ::StridedArray)
    addgrad!(f, β, copy(g0), α, θ)
end

function verify_reg(f::Regularizer, g0::StridedArray, θ::StridedArray, vr::Real, gr::Array)
    @test_approx_eq vr value(f, θ)
    @test_approx_eq gr grad(f, θ)

    for β in [0.0, 0.5, 1.0], α in [0.0, 1.0, 2.5]
        @test_approx_eq β * g0 + α * gr _addgrad(f, β, g0, α, θ)
    end
end



## Specific regularizers

θ = [2., -3., 4., -5., 6.]
g0 = ones(size(θ))
c1 = 0.6
c2 = 0.8

verify_reg(SqrL2Reg(c2),       g0, θ, (c2/2) * sumabs2(θ), c2 * θ)
verify_reg(L1Reg(c1),          g0, θ, c1 * sumabs(θ), c1 * sign(θ))
verify_reg(ElasticReg(c1, c2), g0, θ,
    c1 * sumabs(θ) + (c2/2) * sumabs2(θ), c1 * sign(θ) + c2 * θ)
