import tensorflow as tf


class AdaptiveNoiseCallback(tf.keras.callbacks.Callback):
    """
    Callback that is run after ever training epoch, setting the noise
    """

    def __init__(self, noise_fn_type, noise_scale_factor, ndim_x, ndim_y):
        """
        :param noise_fn_type: either rule_of_thumb or fixed_rate
        :param noise_scale_factor: for fixed_rate the noise std on x and y
        :param ndim_x: dimension of x input
        :param ndim_y: dimension of y input
        """
        super().__init__()

        self.ndim_x = ndim_x
        self.ndim_y = ndim_y

        assert noise_fn_type in ["rule_of_thumb", "fixed_rate"]
        if noise_fn_type == "rule_of_thumb":
            self._adaptive_noise_fn = lambda n, d: noise_scale_factor * (n + 1) ** (-1 / (4 + d))
        elif noise_fn_type == "fixed_rate":
            self._adaptive_noise_fn = lambda n, d: noise_scale_factor

    def on_epoch_begin(self, epoch, logs=None):
        self.model.x_noise_std.assign(self._adaptive_noise_fn(epoch, self.ndim_x))
        self.model.y_noise_std.assign(self._adaptive_noise_fn(epoch, self.ndim_y))
