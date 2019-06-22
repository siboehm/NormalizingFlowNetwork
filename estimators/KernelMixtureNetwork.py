import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from estimators.DistributionLayers import GaussianKernelsLayer
from estimators.MaximumLikelihoodNNEstimator import MaximumLikelihoodNNEstimator

tfd = tfp.distributions


class KernelMixtureNetwork(MaximumLikelihoodNNEstimator):
    def __init__(self, n_dims, n_centers=50, **kwargs):
        dist_layer = GaussianKernelsLayer(n_centers=n_centers, n_dims=n_dims, trainable_scale=True)
        super().__init__(dist_layer, **kwargs)
        self.dist_layer = dist_layer

    @staticmethod
    def build_function(
        n_dims=1,
        n_centers=30,
        hidden_sizes=(16, 16),
        noise_reg=("fixed_rate", 0.0),
        learning_rate=2e-3,
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
