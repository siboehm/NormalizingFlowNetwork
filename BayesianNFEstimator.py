import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python import tf2

if not tf2.enabled():
    import tensorflow.compat.v2 as tf

    tf.enable_v2_behavior()
    assert tf2.enabled()

tfd = tfp.distributions
from DistributionLayers import InverseNormalizingFlowLayer, MeanFieldLayer


class BayesianNFEstimator(tf.keras.Sequential):
    def __init__(
        self,
        n_dims,
        kl_weight_scale=1.0,
        flow_types=("radial", "radial"),
        hidden_sizes=(10,),
        trainable_base_dist=True,
        activation="tanh",
        learning_rate=2e-2,
        trainable_prior=False,
    ):
        """
        A bayesian net parametrizing a normalizing flow distribution
        :param n_dims: The dimension of the output distribution
        :param kl_weight_scale: Scales how much KL(posterior|prior) influence the loss
        :param flow_types: tuple of flows to use
        :param hidden_sizes: size and depth of net
        :param trainable_base_dist: whether to train the base normal dist
        :param trainable_prior: empirical bayes
        """
        dist_layer = InverseNormalizingFlowLayer(
            flow_types=flow_types,
            n_dims=n_dims,
            trainable_base_dist=trainable_base_dist,
        )

        posterior = self._get_posterior_fn()
        prior = self._get_prior_fn(trainable_prior)
        dense_layers = self._get_dense_layers(
            hidden_sizes=hidden_sizes,
            output_size=dist_layer.get_total_param_size(),
            posterior=posterior,
            prior=prior,
            kl_weight_scale=kl_weight_scale,
            activation=activation,
        )

        super().__init__(dense_layers + [dist_layer])

        negative_log_likelihood = lambda y, p_y: -p_y.log_prob(y)
        self.compile(
            optimizer=tf.compat.v2.optimizers.Adam(learning_rate),
            loss=negative_log_likelihood,
        )

    @staticmethod
    def _get_prior_fn(trainable=False):
        def prior_fn(kernel_size, bias_size=0, dtype=None):
            size = kernel_size + bias_size
            layers = [
                tfp.layers.VariableLayer(
                    shape=size, initializer="zeros", dtype=dtype, trainable=trainable
                ),
                MeanFieldLayer(size, scale=1.0, dtype=dtype),
            ]
            return tf.keras.Sequential(layers)

        return prior_fn

    @staticmethod
    def _get_posterior_fn():
        def posterior_fn(kernel_size, bias_size=0, dtype=None):
            size = kernel_size + bias_size
            layers = [
                tfp.layers.VariableLayer(
                    2 * size, initializer="normal", dtype=dtype, trainable=True
                ),
                MeanFieldLayer(size, scale=None, dtype=dtype),
            ]
            return tf.keras.Sequential(layers)

        return posterior_fn

    @staticmethod
    def _get_dense_layers(
        hidden_sizes, output_size, posterior, prior, kl_weight_scale, activation="relu"
    ):
        assert type(hidden_sizes) == tuple
        hidden = [
            tfp.layers.DenseVariational(
                units=size,
                make_posterior_fn=posterior,
                make_prior_fn=prior,
                kl_weight=kl_weight_scale,
                activation=activation,
            )
            for size in hidden_sizes
        ]
        output = tfp.layers.DenseVariational(
            units=output_size,
            make_posterior_fn=posterior,
            make_prior_fn=prior,
            kl_weight=kl_weight_scale,
            activation="linear",
        )
        return hidden + [output]
