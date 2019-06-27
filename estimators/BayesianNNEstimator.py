import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from estimators.DistributionLayers import MeanFieldLayer
from estimators.BaseEstimator import BaseEstimator

tfd = tfp.distributions


class BayesianNNEstimator(BaseEstimator):
    def __init__(
        self,
        dist_layer,
        kl_weight_scale,
        kl_use_exact=True,
        hidden_sizes=(10,),
        activation="tanh",
        learning_rate=3e-2,
        noise_reg=("fixed_rate", 0.0),
        trainable_prior=False,
        map_mode=False,
        prior_scale=1.0,
        random_seed=22,
    ):
        """
        A bayesian net parametrizing a normalizing flow distribution
        :param dist_layer: A Tfp Distribution Lambda Layer that converts the neural net output into a distribution
        :param kl_weight_scale: Scales how much KL(posterior|prior) influences the loss
        :param hidden_sizes: size and depth of net
        :param noise_reg: Tuple with (type_of_reg, scale_factor)
        :param trainable_prior: empirical bayes
        :param map_mode: If true, will use the mean of the posterior instead of a sample. Default False
        :param prior_scale: The scale of the zero centered priors

        A note on kl_weight_scale: Keras calculates the loss per sample and not for the full dataset. Therefore,
        we need to scale the KL(q||p) loss down to a single sample, which means setting kl_weight_scale = 1/n_datapoints
        """
        self.x_noise_std = tf.Variable(initial_value=0.0, dtype=tf.float32, trainable=False)
        self.y_noise_std = tf.Variable(initial_value=0.0, dtype=tf.float32, trainable=False)
        self.map_mode = map_mode

        posterior = self._get_posterior_fn(map_mode=map_mode)
        prior = self._get_prior_fn(trainable_prior, prior_scale)
        dense_layers = self._get_dense_layers(
            hidden_sizes=hidden_sizes,
            output_size=dist_layer.get_total_param_size(),
            posterior=posterior,
            prior=prior,
            kl_weight_scale=kl_weight_scale,
            kl_use_exact=kl_use_exact,
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

    def score(self, x_data, y_data):
        x_data = x_data.astype(np.float32)
        y_data = y_data.astype(np.float32)

        loss = 0
        nll = self._get_neg_log_likelihood()
        posterior_draws = 1 if self.map_mode else 50
        for _ in range(posterior_draws):
            loss += nll(y_data, self.call(x_data, training=False)).numpy().mean()
        return -loss / posterior_draws

    @staticmethod
    def _get_prior_fn(trainable=False, prior_scale=1.0):
        def prior_fn(kernel_size, bias_size=0, dtype=None):
            size = kernel_size + bias_size
            layers = [
                tfp.layers.VariableLayer(
                    shape=size, initializer="zeros", dtype=dtype, trainable=trainable
                ),
                MeanFieldLayer(size, scale=prior_scale, map_mode=False, dtype=dtype),
            ]
            return tf.keras.Sequential(layers)

        return prior_fn

    @staticmethod
    def _get_posterior_fn(map_mode=False):
        def posterior_fn(kernel_size, bias_size=0, dtype=None):
            size = kernel_size + bias_size
            layers = [
                tfp.layers.VariableLayer(
                    size if map_mode else 2 * size,
                    initializer="normal",
                    dtype=dtype,
                    trainable=True,
                ),
                MeanFieldLayer(size, scale=None, map_mode=map_mode, dtype=dtype),
            ]
            return tf.keras.Sequential(layers)

        return posterior_fn

    def _get_dense_layers(
        self,
        hidden_sizes,
        output_size,
        posterior,
        prior,
        kl_weight_scale=1.0,
        kl_use_exact=True,
        activation="relu",
    ):
        assert type(hidden_sizes) == tuple or type(hidden_sizes) == list
        assert kl_weight_scale <= 1.0

        # these values are assigned once fit is called
        normalization = [tf.keras.layers.Lambda(lambda x: (x - self.x_mean) / (self.x_std + 1e-8))]
        noise_reg = [tf.keras.layers.GaussianNoise(self.x_noise_std)]
        hidden = [
            tfp.layers.DenseVariational(
                units=size,
                make_posterior_fn=posterior,
                make_prior_fn=prior,
                kl_weight=kl_weight_scale,
                kl_use_exact=kl_use_exact,
                activation=activation,
            )
            for size in hidden_sizes
        ]
        output = [
            tfp.layers.DenseVariational(
                units=output_size,
                make_posterior_fn=posterior,
                make_prior_fn=prior,
                kl_weight=kl_weight_scale,
                kl_use_exact=kl_use_exact,
                activation="linear",
            )
        ]
        return normalization + noise_reg + hidden + output
