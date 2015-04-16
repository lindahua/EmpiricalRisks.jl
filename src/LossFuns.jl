module LossFuns


import Base.LinAlg: BlasReal

export

    Loss,
    UnivariateLoss,
    MultivariateLoss,

    PredictionModel,
    UnivariatePredictionModel,
    MultivariatePredictionModel,



# source files
include("common.jl")
include("prediction.jl")
include("loss.jl")

end # module
