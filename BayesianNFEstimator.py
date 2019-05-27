import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python import tf2

if not tf2.enabled():
    import tensorflow.compat.v2 as tf

    tf.enable_v2_behavior()
    assert tf2.enabled()

from DistributionLayers import InverseNormalizingFlowLayer, MeanFieldLayer

tfd = tfp.distributions


class BayesianNFEstimator(tf.keras.Sequential):
    def __init__(
        self,
        n_dims,
        kl_norm_const,
        flow_types=("radial", "radial"),
        hidden_sizes=(16, 16),
        trainable_base_dist=True,
        activation="relu",
        trainable_prior=False,
    ):
        posterior = self._get_posterior_fn()
        prior = self._get_prior_fn(trainable_prior)
        dist_layer = InverseNormalizingFlowLayer(
            flow_types=flow_types,
            n_dims=n_dims,
            trainable_base_dist=trainable_base_dist,
        )
        dense_layers = self._get_dense_layers(
            hidden_sizes=hidden_sizes,
            output_size=dist_layer.get_total_param_size(),
            posterior=posterior,
            prior=prior,
            kl_norm_const=kl_norm_const,
            activation=activation,
        )

        super().__init__(dense_layers + [dist_layer])

        negative_log_likelihood = lambda y, p_y: -p_y.log_prob(y)
        self.compile(
            optimizer=tf.compat.v2.optimizers.Adam(0.03), loss=negative_log_likelihood
        )

    @staticmethod
    def _get_prior_fn(trainable=False):
        def prior_fn(kernel_size, bias_size=0, dtype=None):
            size = kernel_size + bias_size
            layers = [
                tfp.layers.VariableLayer(2 * size, dtype=dtype, trainable=trainable),
                MeanFieldLayer(size, uniform_scale=False, dtype=dtype),
            ]
            return tf.keras.Sequential(layers)

        return prior_fn

    @staticmethod
    def _get_posterior_fn():
        def posterior_fn(kernel_size, bias_size=0, dtype=None):
            size = kernel_size + bias_size
            layers = [
                tfp.layers.VariableLayer(2 * size, dtype=dtype, trainable=True),
                MeanFieldLayer(size, uniform_scale=False, dtype=dtype),
            ]
            return tf.keras.Sequential(layers)

        return posterior_fn

    @staticmethod
    def _get_dense_layers(
        hidden_sizes, output_size, posterior, prior, kl_norm_const, activation="relu"
    ):
        assert type(hidden_sizes) == tuple
        hidden = [
            tfp.layers.DenseVariational(
                units=size,
                make_posterior_fn=posterior,
                make_prior_fn=prior,
                kl_weight=1 / kl_norm_const,
                activation=activation,
            )
            for size in hidden_sizes
        ]
        output = tfp.layers.DenseVariational(
            units=output_size,
            make_posterior_fn=posterior,
            make_prior_fn=prior,
            kl_weight=1 / kl_norm_const,
            activation="linear",
        )
        return hidden + [output]
