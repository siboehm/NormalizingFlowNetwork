import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python import tf2

if not tf2.enabled():
    import tensorflow.compat.v2 as tf

    tf.enable_v2_behavior()
    assert tf2.enabled()

tfd = tfp.distributions
import numpy as np


class BaseNFEstimator(tf.keras.Sequential):
    def __init__(self, layers):
        self.x_mean = [0.0]
        self.y_mean = [0.0]
        self.x_std = [1.0]
        self.y_std = [1.0]
        super().__init__(layers)

    def fit(self, x, y, batch_size=None, epochs=None, verbose=1, **kwargs):
        self._assign_data_normalization(x, y)
        super().fit(x, y, batch_size, epochs, verbose, kwargs)

    def _assign_data_normalization(self, x, y):
        self.x_mean = np.mean(x, axis=0, dtype=np.float32)
        self.y_mean = np.mean(y, axis=0, dtype=np.float32)
        self.x_std = np.std(x, axis=0, dtype=np.float32)
        self.y_std = np.std(y, axis=0, dtype=np.float32)

    def _get_neg_log_likelihood(self, y_noise_std):
        y_input_model = self._get_input_model(y_noise_std)
        return lambda y, p_y: -p_y.log_prob(y_input_model(y)) + tf.reduce_sum(
            tf.log(self.y_std)
        )

    def _get_input_model(self, y_noise_std):
        y_input_model = tf.keras.Sequential()
        # add data normalization layer
        y_input_model.add(
            tf.keras.layers.Lambda(
                lambda y: (y - tf.ones_like(y) * self.y_mean) / self.y_std
            )
        )
        if y_noise_std:
            # noise will be switched on during training and switched off otherwise automatically
            y_input_model.add(tf.keras.layers.GaussianNoise(y_noise_std))
        return y_input_model

    def pdf(self, x, y):
        assert x.shape == y.shape
        output = self(x)
        y_circ = (y - tf.ones_like(y) * self.y_mean) / self.y_std
        return output.prob(y_circ) / tf.reduce_prod(self.y_std)

    def log_pdf(self, x, y):
        assert x.shape == y.shape
        output = self(x)
        y_circ = (y - tf.ones_like(y) * self.y_mean) / self.y_std
        return output.log_prob(y_circ) - tf.reduce_sum(tf.log(self.y_std))
