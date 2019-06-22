import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from estimators.DistributionLayers import GaussianKernelsLayer
from estimators.BaseEstimator import BaseEstimator

tfd = tfp.distributions


class KernelMixtureNetwork(BaseEstimator):
    def __init__(
        self,
        n_dims,
        n_centers=50,
        hidden_sizes=(16, 16),
        noise_reg=("fixed_rate", 0.0),
        learning_rate=3e-3,
        activation="relu",
    ):
        self.x_noise_std = tf.Variable(initial_value=0.0, dtype=tf.float32, trainable=False)
        self.y_noise_std = tf.Variable(initial_value=0.0, dtype=tf.float32, trainable=False)

        assert len(noise_reg) == 2

        dist_layer = GaussianKernelsLayer(n_centers=n_centers, n_dims=n_dims, trainable_scale=True)
        dense_layers = self._get_dense_layers(
            hidden_sizes=hidden_sizes,
            output_size=dist_layer.get_total_param_size(),
            activation=activation,
        )

        super().__init__(
            dense_layers + [dist_layer], noise_fn_type=noise_reg[0], noise_scale_factor=noise_reg[1]
        )
        self.dist_layer = dist_layer

        self.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate), loss=self._get_neg_log_likelihood()
        )

    @staticmethod
    def build_function(
        n_dims=1,
        n_centers=30,
        hidden_sizes=(16, 16),
        noise_reg=("fixed_rate", 0.0),
        learning_rate=3e-3,
        activation="relu",
    ):
        tf.keras.backend.clear_session()
        return KernelMixtureNetwork(
            n_dims=n_dims,
            n_centers=n_centers,
            hidden_sizes=hidden_sizes,
            noise_reg=noise_reg,
            learning_rate=learning_rate,
            activation=activation,
        )

    def fit(self, x, y, batch_size=None, epochs=None, verbose=1, **kwargs):
        y_mean = np.mean(y, axis=0, dtype=np.float32)
        y_std = np.std(y, axis=0, dtype=np.float32)
        self.dist_layer.set_center_points((y - y_mean) / y_std)
        super().fit(x=x, y=y, batch_size=batch_size, epochs=epochs, verbose=verbose, **kwargs)

    def _get_dense_layers(self, hidden_sizes, output_size, activation):
        assert type(hidden_sizes) == tuple or type(hidden_sizes) == list
        # the data normalization values are assigned once fit is called
        normalization = [tf.keras.layers.Lambda(lambda x: (x - self.x_mean) / (self.x_std + 1e-8))]
        noise_reg = [tf.keras.layers.GaussianNoise(self.x_noise_std)]
        hidden = [tf.keras.layers.Dense(size, activation=activation) for size in hidden_sizes]
        output = [tf.keras.layers.Dense(output_size, activation="linear")]
        return normalization + noise_reg + hidden + output
