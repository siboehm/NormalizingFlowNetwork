import tensorflow as tf
import tensorflow_probability as tfp
from estimators.BaseEstimator import BaseEstimator

tfd = tfp.distributions


class MaximumLikelihoodNNEstimator(BaseEstimator):
    def __init__(
        self,
        dist_layer,
        hidden_sizes=(16, 16),
        noise_reg=("fixed_rate", 0.0),
        learning_rate=3e-3,
        activation="relu",
        random_seed=22,
    ):
        assert len(noise_reg) == 2

        dense_layers = self._get_dense_layers(
            hidden_sizes=hidden_sizes,
            output_size=dist_layer.get_total_param_size(),
            activation=activation,
        )

        super().__init__(
            dense_layers + [dist_layer],
            noise_fn_type=noise_reg[0],
            noise_scale_factor=noise_reg[1],
            random_seed=random_seed,
        )

        self.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate), loss=self._get_neg_log_likelihood()
        )

    def _get_dense_layers(self, hidden_sizes, output_size, activation):
        assert type(hidden_sizes) == tuple or type(hidden_sizes) == list
        # the data normalization values are assigned once fit is called
        normalization = [tf.keras.layers.Lambda(lambda x: (x - self.x_mean) / (self.x_std + 1e-8))]
        noise_reg = [tf.keras.layers.GaussianNoise(self.x_noise_std)]
        hidden = [tf.keras.layers.Dense(size, activation=activation) for size in hidden_sizes]
        output = [tf.keras.layers.Dense(output_size, activation="linear")]
        return normalization + noise_reg + hidden + output
