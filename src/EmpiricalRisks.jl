module EmpiricalRisks

using ArrayViews

import Base.LinAlg: BlasReal
import Base.LinAlg.BLAS: axpy!, gemv!, gemm!

export

    ## loss.jl

    Loss,
    UnivariateLoss,
    MultivariateLoss,

    AbsLoss,
    SqrLoss,
    QuantileLoss,
    HuberLoss,
    HingeLoss,
    SmoothedHingeLoss,
    LogisticLoss,
    MultiLogisticLoss,

    value,
    deriv,
    grad,
    grad!,
    value_and_deriv,
    value_and_grad,
    value_and_grad!,

    ## prediction.jl

    PredictionModel,
    UnivariatePredictionModel,
    MultivariatePredictionModel,
    LinearPred,
    MvLinearPred,
    AffinePred,
    MvAffinePred,

    nsamples,
    predict,
    add_grad!,
    total_grad,
    total_grad!,
    accum_grad!


# source files
include("common.jl")
include("prediction.jl")
include("loss.jl")

end # module
