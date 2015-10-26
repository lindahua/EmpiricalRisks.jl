
# utilities

half(x::Float64) = x * 0.5
half(x::Float32) = x *0.5f0

nonneg(x::Float64) = x < 0.0 ? 0.0 : x  # this also properly takes care of NaN
nonneg(x::Float32) = x < 0.0f0 ? 0.0f0 : x

macro _checkdims(cond)
    quote
        ($cond) || throw(DimensionMismatch())
    end
end

function axpby!{T<:BlasReal}(a::T, x::StridedVector{T}, b::T, y::StridedVector{T})
    if b == zero(T)
        scale!(y, x, a)
    elseif b == one(T)
        axpy!(a, x, y)
    else
        n = length(x)
        @_checkdims length(y) == n
        @inbounds for i = 1:n
            y[i] = a * x[i] + b * y[i]
        end
    end
    y
end

gets(x::StridedVector, i::Int) = x[i]
gets(x::StridedMatrix, i::Int) = view(x,:,i)
gets{T}(x::StridedArray{T,3}, i::Int) = view(x,:,:,i)

shrink{T<:AbstractFloat}(x::T, t::T) = (x > t ? x - t : x < -t ? x + t : zero(T))
shrink{T<:AbstractFloat}(x::StridedVector{T}, t::T) = T[shrink(v, t) for v in x]

no_op(args...) = nothing
