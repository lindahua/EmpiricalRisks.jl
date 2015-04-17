module LossFuns


import Base.LinAlg: BlasReal

export

    # loss.jl
    Loss,
    UnivariateLoss,
    MultivariateLoss,

    AbsLoss,
    SqrLoss,
    HuberLoss,
    HingeLoss,
    LogisticLoss,
    MultiLogisticLoss,

    value,
    deriv,
    value_and_deriv,

    # prediction.jl
    PredictionModel,
    UnivariatePredictionModel,
    MultivariatePredictionModel



# source files
include("common.jl")
include("prediction.jl")
include("loss.jl")

end # module
