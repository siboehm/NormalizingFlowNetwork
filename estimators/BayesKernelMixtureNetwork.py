import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from estimators.DistributionLayers import GaussianKernelsLayer
from estimators.BayesianNNEstimator import BayesianNNEstimator

tfd = tfp.distributions


class BayesKernelMixtureNetwork(BayesianNNEstimator):
    def __init__(self, n_dims, kl_weight_scale, n_centers=50, **kwargs):
        dist_layer = GaussianKernelsLayer(n_centers=n_centers, n_dims=n_dims, trainable_scale=True)
        super().__init__(dist_layer, kl_weight_scale, **kwargs)
        self.dist_layer = dist_layer

    @staticmethod
    def build_function(
        n_dims,
        kl_weight_scale,
        n_centers=50,
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
        return BayesKernelMixtureNetwork(
            n_dims=n_dims,
            kl_weight_scale=kl_weight_scale,
            n_centers=n_centers,
            kl_use_exact=kl_use_exact,
            hidden_sizes=hidden_sizes,
            activation=activation,
            noise_reg=noise_reg,
            learning_rate=learning_rate,
            trainable_prior=trainable_prior,
            map_mode=map_mode,
            prior_scale=prior_scale,
        )

    def fit(self, x, y, batch_size=None, epochs=None, verbose=1, **kwargs):
        y_mean = np.mean(y, axis=0, dtype=np.float32)
        y_std = np.std(y, axis=0, dtype=np.float32)
        self.dist_layer.set_center_points((y - y_mean) / y_std)
        super().fit(x=x, y=y, batch_size=batch_size, epochs=epochs, verbose=verbose, **kwargs)
