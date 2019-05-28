import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python import tf2
from DistributionLayers import InverseNormalizingFlowLayer

if not tf2.enabled():
    import tensorflow.compat.v2 as tf

    tf.enable_v2_behavior()
    assert tf2.enabled()

tfd = tfp.distributions


class MaximumLikelihoodNFEstimator(tf.keras.Sequential):
    def __init__(
        self,
        n_dims,
        flow_types=("radial", "radial"),
        hidden_sizes=(16, 16),
        trainable_base_dist=True,
        x_noise_std=0.0,
        y_noise_std=0.0,
        learning_rate=3e-3,
        activation="relu",
    ):
        dist_layer = InverseNormalizingFlowLayer(
            flow_types=flow_types,
            n_dims=n_dims,
            trainable_base_dist=trainable_base_dist,
        )
        dense_layers = self._get_dense_layers(
            hidden_sizes=hidden_sizes,
            output_size=dist_layer.get_total_param_size(),
            x_noise_std=x_noise_std,
            activation=activation,
        )

        super().__init__(dense_layers + [dist_layer])

        self.compile(
            optimizer=tf.compat.v2.optimizers.Adam(learning_rate),
            loss=self._get_neg_log_likelihood(y_noise_std),
        )

    @staticmethod
    def _get_neg_log_likelihood(y_noise_std):
        if y_noise_std:
            y_noise_layer = tf.keras.layers.GaussianNoise(y_noise_std)
            return lambda y, p_y: -p_y.log_prob(y_noise_layer(y))
        else:
            return lambda y, p_y: -p_y.log_prob(y)

    @staticmethod
    def _get_dense_layers(
        hidden_sizes, output_size, x_noise_std=0.0, activation="relu"
    ):
        assert type(hidden_sizes) == tuple
        assert x_noise_std >= 0.0
        noise_reg = [tf.keras.layers.GaussianNoise(x_noise_std)] if x_noise_std else []
        hidden = [
            tf.keras.layers.Dense(size, activation=activation) for size in hidden_sizes
        ]
        output = [tf.keras.layers.Dense(output_size, activation="linear")]
        return noise_reg + hidden + output
