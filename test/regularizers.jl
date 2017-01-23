using EmpiricalRisks
using Base.Test

import EmpiricalRisks: shrink


## auxiliary functions

function _val_and_addgrad(f::Regularizer, β::Real, g0::StridedArray, α::Real, θ::StridedArray)
    value_and_addgrad!(f, β, copy(g0), α, θ)
end

function verify_reg(f::Regularizer, g0::StridedArray, θ::StridedArray, vr::Real, gr::Array, pr::Array)
    @test_approx_eq vr value(f, θ)

    (v, g) = value_and_grad(f, θ)
    @test_approx_eq vr v
    @test_approx_eq gr g

    for β in [0.0, 0.5, 1.0], α in [0.0, 1.0, 2.5]
        v_ = α * vr
        g_ = β * g0 + α * gr

        (v, g) = _val_and_addgrad(f, β, g0, α, θ)
        @test_approx_eq v_ v
        @test_approx_eq g_ g
    end

    @test_approx_eq pr prox(f, θ)
end


## shrink

@test shrink(5.0, 2.0) == 3.0
@test shrink(1.0, 2.0) == 0.0
@test shrink(-1.0, 2.0) == 0.0
@test shrink(-5.0, 2.0) == -3.0

## Specific regularizers

θ = [2., -3., 4., -5., 6.]
g0 = ones(size(θ))
c1 = 0.6
c2 = 0.8

verify_reg(SqrL2Reg(c2), g0, θ,
    (c2/2) * sumabs2(θ),
    c2 * θ,
    1.0 / (1.0 + c2) * θ)

@test_approx_eq prox(SqrL2Reg(c2), θ, 1.5) prox(SqrL2Reg(1.5 * c2), θ)


verify_reg(L1Reg(c1), g0, θ,
    c1 * sumabs(θ),
    c1 * sign(θ),
    shrink(θ, c1))

@test_approx_eq prox(L1Reg(c1), θ, 1.5) prox(L1Reg(1.5 * c1), θ)


verify_reg(ElasticReg(c1, c2), g0, θ,
    c1 * sumabs(θ) + (c2/2) * sumabs2(θ),
    c1 * sign(θ) + c2 * θ,
    shrink(1.0 / (1.0 + c2) * θ, c1 / (1.0 + c2)))

@test_approx_eq prox(ElasticReg(c1, c2), θ, 1.5) prox(ElasticReg(1.5 * c1, 1.5 * c2), θ)


# Projection regularizers: NonNegReg, SimplexReg, L1BallReg

reg = NonNegReg()

θ = [0.1, 0.9]
@test value(reg, θ) == 0.
@test_throws ErrorException value_and_addgrad!(reg, 0., similar(θ), 1., θ) 
@test prox!(reg, θ, θ, 1.0) == [0.1, 0.9]

θ = [0.2, -0.5]
@test value(reg, θ) == Inf 
@test prox!(reg, θ, θ, 1.0) == [0.2, 0.]

θ = [0., -1., 4., -0.1]
@test value(reg, θ) == Inf
@test prox!(reg, θ, θ, 1.0) == [0., 0., 4., 0.]

θ = [0. 1. ; -1. 0]
@test value(reg, θ) == Inf  
@test prox!(reg, θ, θ, 1.0) == [0. 1. ; 0. 0.]

θ = [0. 0.5 ; 0.5 0.]
θ1 = Base.view(θ, :, 2)
θ2 = similar(θ1)
@test value(reg, θ1) == 0. 
@test prox!(reg, θ2, θ1, 1.0) == [0.5, 0.]

θ1 = Base.view(θ, 1, :)
θ2 = similar(θ1)
@test value(reg, θ1) == 0.
@test prox!(reg, θ2, θ1, 1.0) == [0., 0.5]



reg = SimplexReg(1.0)

θ = [0.1, 0.9]
@test value(reg, θ) == 0.
@test_throws ErrorException value_and_addgrad!(reg, 0., similar(θ), 1., θ) 
@test prox!(reg, θ, θ, 1.0) == [0.1, 0.9]

θ = [0., 0.]
@test value(reg, θ) == Inf
@test prox!(reg, θ, θ, 1.0) == [0.5, 0.5]

θ = [0., -1., 4., -0.1]
@test value(reg, θ) == Inf  
@test prox!(reg, θ, θ, 1.0) == [0., 0., 1., 0.]

θ = [0. 1. ; 1. 0]
@test value(reg, θ) == Inf
@test prox!(reg, θ, θ, 1.0) == [0. 0.5 ; 0.5 0.]

θ = [0. 0.5 ; 0.5 0.]
θ1 = Base.view(θ, :, 2)
θ2 = similar(θ1)
@test value(reg, θ1) == Inf
@test prox!(reg, θ2, θ1, 1.0) == [0.75, 0.25]

θ1 = Base.view(θ, 1, :)
θ2 = similar(θ1)
@test value(reg, θ1) == Inf
@test prox!(reg, θ2, θ1, 1.0) == [0.25, 0.75]



reg = L1BallReg(1.0)

θ = [0.1, 0.9]
@test value(reg, θ) == 0.
@test_throws ErrorException value_and_addgrad!(reg, 0., similar(θ), 1., θ) 
@test prox!(reg, θ, θ, 1.0) == [0.1, 0.9]

θ = [0.2, -0.5]
@test value(reg, θ) == 0. 
@test prox!(reg, θ, θ, 1.0) == [0.2, -0.5]

θ = [0., -1., 4., -0.1]
@test value(reg, θ) == Inf
@test prox!(reg, θ, θ, 1.0) == [0., 0., 1., 0.]

θ = [0. 1. ; -1. 0]
@test value(reg, θ) == Inf  
@test prox!(reg, θ, θ, 1.0) == [0. 0.5 ; -0.5 0]

θ = [0. 0.5 ; 0.5 0.]
θ1 = Base.view(θ, :, 2)
θ2 = similar(θ1)
@test value(reg, θ1) == 0. 
@test prox!(reg, θ2, θ1, 1.0) == [0.5, 0.]

θ1 = Base.view(θ, 1, :)
θ2 = similar(θ1)
@test value(reg, θ1) == 0.
@test prox!(reg, θ2, θ1, 1.0) == [0., 0.5]
