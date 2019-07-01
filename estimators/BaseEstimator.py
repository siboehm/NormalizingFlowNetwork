import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from estimators.AdaptiveNoiseCallback import AdaptiveNoiseCallback

tfd = tfp.distributions


class BaseEstimator(tf.keras.Sequential):
    def __init__(self, layers, noise_fn_type="fixed_rate", noise_scale_factor=0.0, random_seed=22):
        tf.set_random_seed(random_seed)
        self.noise_fn_type = noise_fn_type
        self.noise_scale_factor = noise_scale_factor

        super().__init__(layers)

    def fit(self, x, y, batch_size=None, epochs=None, verbose=1, **kwargs):
        self._assign_data_normalization(x, y)
        assert len(x.shape) == len(y.shape) == 2
        ndim_x = x.shape[1]
        ndim_y = y.shape[1]
        super().fit(
            x=x,
            y=y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            callbacks=[
                AdaptiveNoiseCallback(self.noise_fn_type, self.noise_scale_factor, ndim_x, ndim_y),
                tf.keras.callbacks.TerminateOnNaN(),
            ],
            **kwargs,
        )

    def score(self, x_data, y_data):
        x_data = x_data.astype(np.float32)
        y_data = y_data.astype(np.float32)
        nll = self._get_neg_log_likelihood()
        return -nll(y_data, self.call(x_data, training=False)).numpy().mean()

    def _assign_data_normalization(self, x, y):
        self.x_mean = np.mean(x, axis=0, dtype=np.float32)
        self.y_mean = np.mean(y, axis=0, dtype=np.float32)
        self.x_std = np.std(x, axis=0, dtype=np.float32)
        self.y_std = np.std(y, axis=0, dtype=np.float32)

    def _get_neg_log_likelihood(self):
        y_input_model = self._get_input_model()
        return lambda y, p_y: -p_y.log_prob(y_input_model(y)) + tf.reduce_sum(
            tf.math.log(self.y_std)
        )

    def _get_input_model(self):
        y_input_model = tf.keras.Sequential()
        # add data normalization layer
        y_input_model.add(
            tf.keras.layers.Lambda(lambda y: (y - tf.ones_like(y) * self.y_mean) / self.y_std)
        )
        # noise will be switched on during training and switched off otherwise automatically
        y_input_model.add(tf.keras.layers.GaussianNoise(self.y_noise_std))
        return y_input_model

    def pdf(self, x, y):
        assert x.shape == y.shape
        output = self(x)
        y_circ = (y - tf.ones_like(y) * self.y_mean) / self.y_std
        return output.prob(y_circ) / tf.reduce_prod(self.y_std)

    def log_pdf(self, x, y):
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        assert x.shape == y.shape

        output = self(x)
        assert output.event_shape == y.shape[-1]

        y_circ = (y - tf.ones_like(y) * self.y_mean) / self.y_std
        return output.log_prob(y_circ) - tf.reduce_sum(tf.math.log(self.y_std))
