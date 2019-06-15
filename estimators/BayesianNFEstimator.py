import tensorflow as tf
import tensorflow_probability as tfp
from estimators.DistributionLayers import InverseNormalizingFlowLayer, MeanFieldLayer
from estimators.BaseNFEstimator import BaseNFEstimator
import numpy as np

tfd = tfp.distributions


class BayesianNFEstimator(BaseNFEstimator):
    def __init__(
        self,
        n_dims,
        kl_weight_scale,
        kl_use_exact=True,
        n_flows=2,
        hidden_sizes=(10,),
        trainable_base_dist=True,
        activation="tanh",
        learning_rate=2e-2,
        noise_reg=("fixed_rate", 0.0),
        trainable_prior=False,
        map_mode=False,
        prior_scale=1.0,
    ):
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
        self.x_noise_std = tf.Variable(initial_value=0.0, dtype=tf.float32, trainable=False)
        self.y_noise_std = tf.Variable(initial_value=0.0, dtype=tf.float32, trainable=False)

        dist_layer = InverseNormalizingFlowLayer(
            flow_types=["radial"] * n_flows, n_dims=n_dims, trainable_base_dist=trainable_base_dist
        )

        posterior = self._get_posterior_fn(map_mode=map_mode)
        prior = self._get_prior_fn(trainable_prior, prior_scale)
        dense_layers = self._get_dense_layers(
            hidden_sizes=hidden_sizes,
            output_size=dist_layer.get_total_param_size(),
            posterior=posterior,
            prior=prior,
            kl_weight_scale=kl_weight_scale,
            kl_use_exact=kl_use_exact,
            activation=activation,
        )

        super().__init__(
            dense_layers + [dist_layer], noise_fn_type=noise_reg[0], noise_scale_factor=noise_reg[1]
        )

        self.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate), loss=self._get_neg_log_likelihood()
        )

    @staticmethod
    def build_function(
        n_dims=1,
        kl_weight_scale=1.0,
        kl_use_exact=True,
        n_flows=2,
        hidden_sizes=(10,),
        trainable_base_dist=True,
        activation="tanh",
        noise_reg=("fixed_rate", 0.0),
        learning_rate=2e-2,
        trainable_prior=False,
        prior_scale=1.0,
    ):
        # this is necessary, else there'll be processes hanging around hogging memory
        tf.keras.backend.clear_session()
        return BayesianNFEstimator(
            n_dims=n_dims,
            kl_weight_scale=kl_weight_scale,
            kl_use_exact=kl_use_exact,
            n_flows=n_flows,
            hidden_sizes=hidden_sizes,
            trainable_base_dist=trainable_base_dist,
            activation=activation,
            noise_reg=noise_reg,
            learning_rate=learning_rate,
            trainable_prior=trainable_prior,
            prior_scale=prior_scale,
        )

    @staticmethod
    def _get_prior_fn(trainable=False, prior_scale=1.0):
        def prior_fn(kernel_size, bias_size=0, dtype=None):
            size = kernel_size + bias_size
            layers = [
                tfp.layers.VariableLayer(
                    shape=size, initializer="zeros", dtype=dtype, trainable=trainable
                ),
                MeanFieldLayer(size, scale=prior_scale, map_mode=False, dtype=dtype),
            ]
            return tf.keras.Sequential(layers)

        return prior_fn

    @staticmethod
    def _get_posterior_fn(map_mode=False):
        def posterior_fn(kernel_size, bias_size=0, dtype=None):
            size = kernel_size + bias_size
            layers = [
                tfp.layers.VariableLayer(
                    size if map_mode else 2 * size, initializer="zeros", dtype=dtype, trainable=True
                ),
                MeanFieldLayer(size, scale=None, map_mode=map_mode, dtype=dtype),
            ]
            return tf.keras.Sequential(layers)

        return posterior_fn

    def _get_dense_layers(
        self,
        hidden_sizes,
        output_size,
        posterior,
        prior,
        kl_weight_scale=1.0,
        kl_use_exact=True,
        activation="relu",
    ):
        assert type(hidden_sizes) == tuple or type(hidden_sizes) == list
        assert kl_weight_scale <= 1.0

        # these values are assigned once fit is called
        normalization = [tf.keras.layers.Lambda(lambda x: (x - self.x_mean) / (self.x_std + 1e-8))]
        noise_reg = [tf.keras.layers.GaussianNoise(self.x_noise_std)]
        hidden = [
            tfp.layers.DenseVariational(
                units=size,
                make_posterior_fn=posterior,
                make_prior_fn=prior,
                kl_weight=kl_weight_scale,
                kl_use_exact=kl_use_exact,
                activation=activation,
            )
            for size in hidden_sizes
        ]
        output = [
            tfp.layers.DenseVariational(
                units=output_size,
                make_posterior_fn=posterior,
                make_prior_fn=prior,
                kl_weight=kl_weight_scale,
                kl_use_exact=kl_use_exact,
                activation="linear",
            )
        ]
        return normalization + noise_reg + hidden + output
