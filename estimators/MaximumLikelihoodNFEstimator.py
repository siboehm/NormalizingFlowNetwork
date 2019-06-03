import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
from estimators.DistributionLayers import InverseNormalizingFlowLayer
from estimators.BaseNFEstimator import BaseNFEstimator


class MaximumLikelihoodNFEstimator(BaseNFEstimator):
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
    def build_function(
        n_dims=1,
        flow_types=("radial", "radial"),
        hidden_sizes=(16, 16),
        trainable_base_dist=True,
        x_noise_std=0.0,
        y_noise_std=0.0,
        learning_rate=3e-3,
        activation="tanh",
    ):
        return MaximumLikelihoodNFEstimator(
            n_dims=n_dims,
            flow_types=flow_types,
            hidden_sizes=hidden_sizes,
            trainable_base_dist=trainable_base_dist,
            x_noise_std=x_noise_std,
            y_noise_std=y_noise_std,
            learning_rate=learning_rate,
            activation=activation,
        )

    def _get_dense_layers(self, hidden_sizes, output_size, x_noise_std, activation):
        assert type(hidden_sizes) == tuple
        assert x_noise_std >= 0.0

        # these values are assigned once fit is called
        normalization = [
            tf.keras.layers.Lambda(lambda x: (x - self.x_mean) / (self.x_std + 1e-8))
        ]
        noise_reg = [tf.keras.layers.GaussianNoise(x_noise_std)]
        hidden = [
            tf.keras.layers.Dense(size, activation=activation) for size in hidden_sizes
        ]
        output = [tf.keras.layers.Dense(output_size, activation="linear")]
        return normalization + noise_reg + hidden + output
