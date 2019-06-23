from estimators.BayesNormalizingFlowNetwork import BayesNormalizingFlowNetwork
from estimators.BayesMixtureDensityNetwork import BayesMixtureDensityNetwork
from estimators.BayesKernelMixtureNetwork import BayesKernelMixtureNetwork
from estimators.NormalizingFlowNetwork import NormalizingFlowNetwork
from estimators.MixtureDensityNetwork import MixtureDensityNetwork
from estimators.KernelMixtureNetwork import KernelMixtureNetwork

ESTIMATORS = {
    "bayesian_NFN": BayesNormalizingFlowNetwork,
    "bayesian_KMN": BayesKernelMixtureNetwork,
    "bayesian_MDN": BayesMixtureDensityNetwork,
    "NFN": NormalizingFlowNetwork,
    "KMN": KernelMixtureNetwork,
    "MDN": MixtureDensityNetwork,
}
