import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions


class BaseEstimator(tf.keras.Sequential):
    def __init__(self, layers, noise_fn_type="fixed_rate", noise_scale_factor=0.0, random_seed=22):
        tf.set_random_seed(random_seed)
        self.noise_fn_type = noise_fn_type
        self.noise_scale_factor = noise_scale_factor

        super().__init__(layers)

    def fit(self, x, y, batch_size=None, epochs=None, verbose=1, **kwargs):
        self._assign_data_normalization(x, y)
        self._assign_noise_regularisation(n_dims=x.shape[1] + y.shape[1], n_datapoints=x.shape[0])
        assert len(x.shape) == len(y.shape) == 2
        super().fit(
            x=x,
            y=y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            callbacks=[tf.keras.callbacks.TerminateOnNaN()],
            **kwargs,
        )

    def _assign_noise_regularisation(self, n_dims, n_datapoints):
        assert self.noise_fn_type in ["rule_of_thumb", "fixed_rate"]
        if self.noise_fn_type == "rule_of_thumb":
            noise_std = self.noise_scale_factor * (n_datapoints + 1) ** (-1 / (4 + n_dims))
            self.x_noise_std.assign(noise_std)
            self.y_noise_std.assign(noise_std)
        elif self.noise_fn_type == "fixed_rate":
            self.x_noise_std.assign(self.noise_scale_factor)
            self.y_noise_std.assign(self.noise_scale_factor)

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
