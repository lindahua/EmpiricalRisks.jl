
### Generic functions

logistic{T<:BlasReal}(x::T) = one(T) ./ (one(T) .+ exp(-x))

### Links

## LogitLink

immutable LogitLink{P<:PredictionModel{1,0}} <: PredictionModel{1,0}
    predmodel::P

    LogitLink(predmodel::LinearPred) = new(predmodel)
    LogitLink(predmodel::AffinePred) = new(predmodel)
end

LogitLink{M<:PredictionModel{1,0}}(predmodel::M) = LogitLink{M}(predmodel)

function predict{T<:BlasReal,M<:PredictionModel{1,0}}(lm::LogitLink{M}, θ::StridedVector{T}, x::StridedVector{T})
    p = predict(lm.predmodel, θ, x)
    logistic(p)
end

function predict{T<:BlasReal,M<:PredictionModel{1,0}}(lm::LogitLink{M}, θ::StridedVector{T}, X::StridedMatrix{T})
    P = predict(lm.predmodel, θ, X)
    broadcast(logistic, P)
end

function predict!{T<:BlasReal,M<:PredictionModel{1,0}}(r::StridedVector{T}, lm::LogitLink{M}, θ::StridedVector{T}, X::StridedMatrix{T})
    predict!(r, lm.predmodel, θ, X)
    broadcast!(logistic, r, r)
end

for link = (:(LogitLink),)
    for op = (:(inputlen),:(inputsize),:(outputlen),:(outputsize),:(paramlen),:(paramsize))
        eval(:(($op{M<:PredictionModel{1,0}})(lm::$(link){M}) = ($op)(lm.predmodel)))
    end
    eval(:((ninputs{M<:PredictionModel{1,0}})(lm::$(link){M}, x::StridedVecOrMat) = (ninputs)(lm.predmodel, x)))
end
