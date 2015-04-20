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
    SumLoss,
    SumSqrLoss,
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

    inputlen,
    inputsize,
    outputlen,
    outputsize,
    paramlen,
    paramsize,
    ninputs,
    isvalidparam,
    predict,

    # risks

    RiskModel,
    SupervisedRiskModel,
    riskmodel,
    risk,
    addgrad!,

    # regularizers

    Regularizer,
    SqrL2Reg,
    L1Reg,
    ElasticReg


# source files
include("common.jl")
include("prediction.jl")
include("loss.jl")
include("risks.jl")
include("regularizer.jl")

end # module
