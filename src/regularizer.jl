# Regularizer

abstract Regularizer

## generic methods

value_and_grad{T<:AbstractFloat}(f::Regularizer, θ::StridedArray{T}) = value_and_addgrad!(f, zero(T), similar(θ), one(T), θ)

prox!(f::Regularizer, θ::StridedArray) = prox!(f, θ, θ)

prox(f::Regularizer, θ::StridedArray) = prox!(f, similar(θ), θ, 1.0)
prox(f::Regularizer, θ::StridedArray, λ::Real) = prox!(f, similar(θ), θ, λ)

function value_and_grad!{T<:AbstractFloat,N}(rm::RiskModel, reg::Regularizer, g::StridedArray{T,N}, θ::StridedArray{T,N}, X, y)
    v_risk, _ = value_and_addgrad!(rm, zero(T), g, one(T), θ, X, y)
    v_regr, _ = value_and_addgrad!(reg, one(T), g, one(T), θ)
    return (v_risk + v_regr, g)
end


## ZeroReg: 0

immutable ZeroReg <: Regularizer
end

value{T<:Real}(::ZeroReg, θ::StridedArray{T}) = zero(T)

value_and_addgrad!{T<:Real,N}(::ZeroReg, β::T, g::StridedArray{T,N}, α::T, θ::StridedArray{T,N}) =
    (zero(T), g)

prox!{T<:Real}(::ZeroReg, r::StridedArray{T}, θ::StridedArray{T}, λ::Real) =
    copy!(r, θ)


## SqrL2Reg: (c/2) * ||θ||_2^2

immutable SqrL2Reg{T<:AbstractFloat} <: Regularizer
    c::T
end

SqrL2Reg{T<:AbstractFloat}(c::T) = SqrL2Reg{T}(c)

value{T<:BlasReal}(f::SqrL2Reg{T}, θ::StridedArray{T}) = half(f.c * sumabs2(θ))

function value_and_addgrad!{T<:BlasReal,N}(f::SqrL2Reg{T}, β::T, g::StridedArray{T,N}, α::T, θ::StridedArray{T,N})
    @_checkdims length(g) == length(θ)
    c = f.c * α
    v = zero(T)
    if β == zero(T)
        @inbounds for i in eachindex(θ)
            θ_i = θ[i]
            v += abs2(θ_i)
            g[i] = c * θ_i
        end
    else
        @inbounds for i in eachindex(θ)
            θ_i = θ[i]
            v += abs2(θ_i)
            g[i] = β * g[i] + c * θ_i
        end
    end
    (half(c * v), g)
end

function prox!{T<:BlasReal}(f::SqrL2Reg{T}, r::StridedArray{T}, θ::StridedArray{T}, λ::Real)
    c = convert(T, λ) * f.c
    scale!(r, θ, one(T) / (one(T) + c))
end


## L1Reg: c * ||θ||_1

immutable L1Reg{T<:AbstractFloat} <: Regularizer
    c::T
end

L1Reg{T<:AbstractFloat}(c::T) = L1Reg{T}(c)

value{T<:BlasReal}(f::L1Reg{T}, θ::StridedArray{T}) = f.c * sumabs(θ)

function value_and_addgrad!{T<:BlasReal,N}(f::L1Reg{T}, β::T, g::StridedArray{T,N}, α::T, θ::StridedArray{T,N})
    @_checkdims length(g) == length(θ)
    c = f.c * α
    v = zero(T)
    if β == zero(T)
        for i in eachindex(θ)
            θ_i = θ[i]
            v += abs(θ_i)
            @inbounds g[i] = c * sign(θ_i)
        end
    else
        for i in eachindex(θ)
            θ_i = θ[i]
            v += abs(θ_i)
            @inbounds g[i] = β * g[i] + c * sign(θ_i)
        end
    end
    (c * v, g)
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

immutable ElasticReg{T<:AbstractFloat} <: Regularizer
    c1::T
    c2::T
end

ElasticReg{T<:AbstractFloat}(c1::T, c2::T) = ElasticReg{T}(c1, c2)

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

function value_and_addgrad!{T<:BlasReal,N}(f::ElasticReg{T}, β::T, g::StridedArray{T,N}, α::T, θ::StridedArray{T,N})
    @_checkdims length(g) == length(θ)
    c1 = f.c1 * α
    c2 = f.c2 * α

    v1 = zero(T)
    v2 = zero(T)
    if β == zero(T)
        @inbounds for i in eachindex(θ)
            θ_i = θ[i]
            v1 += abs(θ_i)
            v2 += abs2(θ_i)
            g[i] = c1 * sign(θ_i) + c2 * θ_i
        end
    else
        @inbounds for i in eachindex(θ)
            θ_i = θ[i]
            v1 += abs(θ_i)
            v2 += abs2(θ_i)
            g[i] = β * g[i] + (c1 * sign(θ_i) + c2 * θ_i)
        end
    end
    (c1 * v1 + half(c2 * v2), g)
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




## NonNegReg: θ[i] >= 0 ∀ i 
immutable NonNegReg <: Regularizer
end

function value{T<:Real}(::NonNegReg, θ::StridedArray{T})
    s = zero(T)
    @inbounds for i in eachindex(θ)
        θ[i] < s && return Inf      # value = + Inf if outside domain
    end
    s   # value = 0. otherwise
end

value_and_addgrad!{T<:Real,N}(::NonNegReg, β::T, g::StridedArray{T,N}, α::T, θ::StridedArray{T,N}) =
    error("Gradient undefined for the NonNegReg regularizer")

#  NOTE : λ is ignored
function prox!{T<:Real}(::NonNegReg, r::StridedArray{T}, θ::StridedArray{T}, λ::Real)
    @_checkdims size(r) == size(θ)
    c = convert(T, λ)
    s = zero(T)
    for i in eachindex(θ)
        r[i] = θ[i] < s ? s : θ[i]
    end
    r    
end


## Simplex: Σ θ[i] = c, θ[i] >= 0 ∀ i 
immutable SimplexReg{T<:AbstractFloat} <: Regularizer
    c::T
end

SimplexReg{T<:AbstractFloat}(c::T=1.0) = SimplexReg{T}(c)

function value{T<:BlasReal}(f::SimplexReg{T}, θ::StridedArray{T})
    z = zero(T)
    s = zero(T)
    @inbounds for i in eachindex(θ)
        θ[i] < z && return Inf       # value = + Inf if outside domain
        s += θ[i]
    end

    return abs(s - f.c) <= 1.0e-12 ? z : Inf
end

value_and_addgrad!{T<:BlasReal,N}(f::SimplexReg{T}, β::T, g::StridedArray{T,N}, α::T, θ::StridedArray{T,N}) =
    error("Gradient undefined for the SimplexReg regularizer")

# Source = Efficient Projections on the l1-Ball for Learning in High Dimensions - J.Duchi
# Note : λ is ignored
function prox!{T<:BlasReal}(f::SimplexReg{T}, r::StridedArray{T}, θ::StridedArray{T}, λ::Real)
    @_checkdims size(r) == size(θ)

    z  = zero(T)
    ρ, s = 0, z
    U  = [1:length(θ);]
    cU = length(θ)

    # pivot search by divide and conquer
    while cU > 0
        k  = U[rand(1:cU)]  # pick an element at random
        vk = θ[k]
        dρ, ds = 0, z
        @inbounds for i in 1:cU
            j = U[i]
            θ[j] >= vk || continue
            dρ += 1
            ds += θ[j]
        end            

        if (s+ds) - (ρ+dρ)*vk < f.c
            s += ds
            ρ += dρ
            nCu = 0
            @inbounds for i in 1:cU
                j = U[i]
                θ[j] < vk || continue
                nCu += 1 
                U[nCu] = U[i]
            end
            cU = nCu
        else
            nCu = 0
            @inbounds for i in 1:cU
                j = U[i]
                θ[j] >= vk || continue
                j != k     || continue
                nCu += 1 
                U[nCu] = U[i]
            end
            cU = nCu
        end
    end

    # apply pivot
    @inbounds for i in eachindex(θ)
        r[i]  = max( θ[i] - (s - f.c)/ρ, z )
    end
    r

    r
end


## L1Ball: Σ |θ[i]| = c
immutable L1BallReg{T<:AbstractFloat} <: Regularizer
    c::T
end

L1BallReg{T<:AbstractFloat}(c::T) = L1BallReg{T}(c)

function value{T<:BlasReal}(f::L1BallReg{T}, θ::StridedArray{T})
    z = zero(T)
    s = zero(T)
    @inbounds for i in eachindex(θ)
        s += abs(θ[i])
        s > f.c && return Inf
    end

    return z
end

value_and_addgrad!{T<:BlasReal,N}(f::L1BallReg{T}, β::T, g::StridedArray{T,N}, α::T, θ::StridedArray{T,N}) =
    error("Gradient undefined for the L1BallReg regularizer")

# Source = Efficient Projections on the l1-Ball for Learning in High Dimensions - J.Duchi
# Note : λ is ignored
function prox!{T<:BlasReal}(f::L1BallReg{T}, r::StridedArray{T}, θ::StridedArray{T}, λ::Real)
    @_checkdims size(r) == size(θ)

    z  = zero(T)
    ρ, s = 0, z
    U  = [1:length(θ);]
    cU = length(θ)

    # early exit if within limits
    if sumabs(θ) <= f.c
        copy!(r, θ)
        return r
    end

    # pivot search by divide and conquer
    while cU > 0
        k  = U[rand(1:cU)]
        vk = abs(θ[k])
        dρ, ds = 0, z
        @inbounds for i in 1:cU
            j = U[i]
            abs(θ[j]) >= vk || continue
            dρ += 1
            ds += abs(θ[j])
        end            

        if (s+ds) - (ρ+dρ)*vk < f.c
            s += ds
            ρ += dρ
            nCu = 0
            @inbounds for i in 1:cU
                j = U[i]
                abs(θ[j]) < vk || continue
                nCu += 1 
                U[nCu] = U[i]
            end
            cU = nCu
        else
            nCu = 0
            @inbounds for i in 1:cU
                j = U[i]
                abs(θ[j]) >= vk || continue
                j != k || continue
                nCu += 1 
                U[nCu] = U[i]
            end
            cU = nCu
        end
    end

    # apply pivot
    @inbounds for i in eachindex(θ)
        r[i]  = sign(θ[i]) * max( abs(θ[i]) - (s - f.c)/ρ, z )
    end
    r
end
