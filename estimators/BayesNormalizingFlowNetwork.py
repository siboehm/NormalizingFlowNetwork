import tensorflow as tf
import tensorflow_probability as tfp
from estimators.DistributionLayers import InverseNormalizingFlowLayer
from estimators.BayesianNNEstimator import BayesianNNEstimator

tfd = tfp.distributions


class BayesNormalizingFlowNetwork(BayesianNNEstimator):
    def __init__(self, n_dims, kl_weight_scale, n_flows=2, trainable_base_dist=True, **kwargs):
        """
        A bayesian net parametrizing a normalizing flow distribution
        :param n_dims: The dimension of the output distribution
        :param kl_weight_scale: Scales how much KL(posterior|prior) influences the loss
        :param n_flows: The number of flows to use
        :param hidden_sizes: size and depth of net
        :param trainable_base_dist: whether to train the base normal dist
        :param noise_reg: Tuple with (type_of_reg, scale_factor)
        :param trainable_prior: empirical bayes
        :param map_mode: If true, will use the mean of the posterior instead of a sample. Default False
        :param prior_scale: The scale of the zero centered priors

        A note on kl_weight_scale: Keras calculates the loss per sample and not for the full dataset. Therefore,
        we need to scale the KL(q||p) loss down to a single sample, which means setting kl_weight_scale = 1/n_datapoints
        """
        dist_layer = InverseNormalizingFlowLayer(
            flow_types=["radial"] * n_flows, n_dims=n_dims, trainable_base_dist=trainable_base_dist
        )
        super().__init__(dist_layer, kl_weight_scale, **kwargs)

    @staticmethod
    def build_function(
        n_dims,
        kl_weight_scale,
        n_flows=2,
        trainable_base_dist=True,
        kl_use_exact=True,
        hidden_sizes=(10,),
        activation="tanh",
        noise_reg=("fixed_rate", 0.0),
        learning_rate=2e-2,
        trainable_prior=False,
        map_mode=False,
        prior_scale=1.0,
    ):
        # this is necessary, else there'll be processes hanging around hogging memory
        tf.keras.backend.clear_session()
        return BayesNormalizingFlowNetwork(
            n_dims=n_dims,
            kl_weight_scale=kl_weight_scale,
            n_flows=n_flows,
            trainable_base_dist=trainable_base_dist,
            kl_use_exact=kl_use_exact,
            hidden_sizes=hidden_sizes,
            activation=activation,
            noise_reg=noise_reg,
            learning_rate=learning_rate,
            trainable_prior=trainable_prior,
            map_mode=map_mode,
            prior_scale=prior_scale,
        )
