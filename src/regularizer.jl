# Regularizer

abstract Regularizer

## generic methods

grad{T<:FloatingPoint}(f::Regularizer, θ::StridedArray{T}) = addgrad!(f, zero(T), similar(θ), one(T), θ)

prox!(f::Regularizer, θ::StridedArray) = prox!(f, θ, θ)

prox(f::Regularizer, θ::StridedArray) = prox!(f, similar(θ), θ, 1.0)
prox(f::Regularizer, θ::StridedArray, λ::Real) = prox!(f, similar(θ), θ, λ)


## SqrL2Reg: (c/2) * ||θ||_2^2

immutable SqrL2Reg{T<:FloatingPoint} <: Regularizer
    c::T
end

SqrL2Reg{T<:FloatingPoint}(c::T) = SqrL2Reg{T}(c)

value{T<:BlasReal}(f::SqrL2Reg{T}, θ::StridedArray{T}) = half(f.c * sumabs2(θ))

function addgrad!{T<:BlasReal,N}(f::SqrL2Reg{T}, β::T, g::StridedArray{T,N}, α::T, θ::StridedArray{T,N})
    axpby!(f.c * α, θ, β, g)
end

function prox!{T<:BlasReal}(f::SqrL2Reg{T}, r::StridedArray{T}, θ::StridedArray{T}, λ::Real)
    c = convert(T, λ) * f.c
    scale!(r, θ, one(T) / (one(T) + c))
end


## L1Reg: c * ||θ||_1

immutable L1Reg{T<:FloatingPoint} <: Regularizer
    c::T
end

L1Reg{T<:FloatingPoint}(c::T) = L1Reg{T}(c)

value{T<:BlasReal}(f::L1Reg{T}, θ::StridedArray{T}) = f.c * sumabs(θ)

function addgrad!{T<:BlasReal,N}(f::L1Reg{T}, β::T, g::StridedArray{T,N}, α::T, θ::StridedArray{T,N})
    @_checkdims length(g) == length(θ)
    c = f.c * α
    if β == zero(T)
        for i in eachindex(θ)
            @inbounds g[i] = c * sign(θ[i])
        end
    else
        for i in eachindex(θ)
            @inbounds g[i] = β * g[i] + c * sign(θ[i])
        end
    end
    g
end

function prox!{T<:BlasReal}(f::L1Reg{T}, r::StridedArray{T}, θ::StridedArray{T}, λ::Real)
    @_checkdims size(r) == size(θ)
    c = convert(T, λ) * f.c
    for i in eachindex(θ)
        r[i] = shrink(θ[i], c)
    end
    r
end


## ElasticNet: c1 * ||θ||_1 + c2 * ||θ||_2^2

immutable ElasticReg{T<:FloatingPoint} <: Regularizer
    c1::T
    c2::T
end

ElasticReg{T<:FloatingPoint}(c1::T, c2::T) = ElasticReg{T}(c1, c2)

function value{T<:BlasReal}(f::ElasticReg{T}, θ::StridedArray{T})
    s = zero(T)
    c1 = f.c1
    c2_h = half(f.c2)
    @inbounds for i in eachindex(θ)
        θ_i = θ[i]
        s += c1 * abs(θ_i) + c2_h * abs2(θ_i)
    end
    s
end

function addgrad!{T<:BlasReal,N}(f::ElasticReg{T}, β::T, g::StridedArray{T,N}, α::T, θ::StridedArray{T,N})
    @_checkdims length(g) == length(θ)
    c1 = f.c1 * α
    c2 = f.c2 * α

    if β == zero(T)
        @inbounds for i in eachindex(θ)
            θ_i = θ[i]
            g[i] = c1 * sign(θ_i) + c2 * θ_i
        end
    else
        @inbounds for i in eachindex(θ)
            θ_i = θ[i]
            g[i] = β * g[i] + (c1 * sign(θ_i) + c2 * θ_i)
        end
    end
    g
end

function prox!{T<:BlasReal}(f::ElasticReg{T}, r::StridedArray{T}, θ::StridedArray{T}, λ::Real)
    @_checkdims size(r) == size(θ)
    c1 = convert(T, λ) * f.c1
    c2 = convert(T, λ) * f.c2
    c = one(T) / (one(T) + c2)
    t = c1 / (one(T) + c2)
    @inbounds for i in eachindex(θ)
        r[i] = shrink(c * θ[i], t)
    end
    r
end
