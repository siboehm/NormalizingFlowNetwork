import tensorflow as tf
import tensorflow_probability as tfp
from estimators.DistributionLayers import GaussianMixtureLayer
from estimators.MaximumLikelihoodNNEstimator import MaximumLikelihoodNNEstimator

tfd = tfp.distributions


class MixtureDensityNetwork(MaximumLikelihoodNNEstimator):
    def __init__(self, n_dims, n_centers, **kwargs):
        dist_layer = GaussianMixtureLayer(n_centers=n_centers, n_dims=n_dims)
        super().__init__(dist_layer, **kwargs)

    @staticmethod
    def build_function(
        n_dims=1,
        n_centers=5,
        hidden_sizes=(16, 16),
        noise_reg=("fixed_rate", 0.0),
        learning_rate=2e-3,
        activation="relu",
    ):
        tf.keras.backend.clear_session()
        return MixtureDensityNetwork(
            n_dims=n_dims,
            n_centers=n_centers,
            hidden_sizes=hidden_sizes,
            noise_reg=noise_reg,
            learning_rate=learning_rate,
            activation=activation,
        )
