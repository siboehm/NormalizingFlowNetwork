import tensorflow as tf
import tensorflow_probability as tfp
from estimators.DistributionLayers import GaussianMixtureLayer
from estimators.BayesianNNEstimator import BayesianNNEstimator

tfd = tfp.distributions


class BayesMixtureDensityNetwork(BayesianNNEstimator):
    def __init__(self, n_dims, kl_weight_scale, n_centers=5, **kwargs):
        dist_layer = GaussianMixtureLayer(n_centers=n_centers, n_dims=n_dims)
        super().__init__(dist_layer, kl_weight_scale, **kwargs)

    @staticmethod
    def build_function(
            n_dims,
            kl_weight_scale,
            n_centers=5,
            kl_use_exact=True,
            hidden_sizes=(10,),
            trainable_base_dist=True,
            activation="tanh",
            noise_reg=("fixed_rate", 0.0),
            learning_rate=2e-2,
            trainable_prior=False,
            map_mode=False,
            prior_scale=1.0,
    ):
        # this is necessary, else there'll be processes hanging around hogging memory
        tf.keras.backend.clear_session()
        return BayesMixtureDensityNetwork(
            n_dims=n_dims,
            kl_weight_scale=kl_weight_scale,
            n_centers=n_centers,
            kl_use_exact=kl_use_exact,
            hidden_sizes=hidden_sizes,
            trainable_base_dist=trainable_base_dist,
            activation=activation,
            noise_reg=noise_reg,
            learning_rate=learning_rate,
            trainable_prior=trainable_prior,
            map_mode=map_mode,
            prior_scale=prior_scale,
        )
