module EmpiricalRisks

using ArrayViews
using Compat

import Base.LinAlg: BlasReal
import Base.LinAlg.BLAS: axpy!, gemv!, gemm!

export

    ## common.jl
    no_op,
    shrink,

    ## loss.jl

    Loss,
    UnivariateLoss,
    MultivariateLoss,

    AbsLoss,
    SqrLoss,
    QuantileLoss,
    EpsilonInsLoss,
    HuberLoss,
    HingeLoss,
    SqrHingeLoss,
    SmoothedHingeLoss,
    LogisticLoss,
    SumLoss,
    SumSqrLoss,
    MultiLogisticLoss,

    value,
    value!,
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
    predict!,

    # risks

    RiskModel,
    SupervisedRiskModel,
    riskmodel,
    value_and_addgrad!,

    # regularizers

    Regularizer,
    ZeroReg,
    SqrL2Reg,
    L1Reg,
    ElasticReg,
    NonNegReg,
    SimplexReg,
    L1BallReg,

    prox, prox!


# source files
include("common.jl")
include("prediction.jl")
include("loss.jl")
include("risks.jl")
include("regularizer.jl")

end # module
