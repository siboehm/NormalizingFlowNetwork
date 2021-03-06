import tensorflow as tf
import tensorflow_probability as tfp
from estimators.DistributionLayers import InverseNormalizingFlowLayer
from estimators.MaximumLikelihoodNNEstimator import MaximumLikelihoodNNEstimator

tfd = tfp.distributions


class NormalizingFlowNetwork(MaximumLikelihoodNNEstimator):
    def __init__(self, n_dims, n_flows=10, trainable_base_dist=True, **kwargs):
        """
        :param n_dims: Dimensionsion of Y. The dimension of X is automatically inferred from the data
        :param n_flows: The number of radial flows to use.
        :param trainable_base_dist: Whether the standard normal base dist has trainable mean + diag. convariance
        """
        dist_layer = InverseNormalizingFlowLayer(
            flow_types=["radial"] * n_flows, n_dims=n_dims, trainable_base_dist=trainable_base_dist
        )
        super().__init__(dist_layer, **kwargs)

    @staticmethod
    def build_function(
        n_dims=1,
        n_flows=3,
        hidden_sizes=(16, 16),
        trainable_base_dist=True,
        noise_reg=("fixed_rate", 0.0),
        learning_rate=3e-3,
        activation="tanh",
    ):
        # this is necessary, else there'll be processes hanging around hogging memory
        tf.keras.backend.clear_session()
        return NormalizingFlowNetwork(
            n_dims=n_dims,
            n_flows=n_flows,
            hidden_sizes=hidden_sizes,
            trainable_base_dist=trainable_base_dist,
            noise_reg=noise_reg,
            learning_rate=learning_rate,
            activation=activation,
        )
