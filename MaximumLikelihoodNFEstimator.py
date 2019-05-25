import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python import tf2
from DistributionLayers import InverseNormalizingFlowLayer

if not tf2.enabled():
    import tensorflow.compat.v2 as tf

    tf.enable_v2_behavior()
    assert tf2.enabled()

tfd = tfp.distributions


class MaximumLikelihoodNFEstimator(tf.keras.Sequential):
    def __init__(
        self,
        n_dims,
        flow_types=("radial", "radial"),
        hidden_sizes=(16, 16),
        trainable_base_dist=True,
        activation="relu",
    ):
        assert type(hidden_sizes) == tuple
        dist_layer = InverseNormalizingFlowLayer(
            flow_types=flow_types,
            n_dims=n_dims,
            trainable_base_dist=trainable_base_dist,
        )
        dense_layers = self._get_dense_layers(
            hidden_sizes=hidden_sizes,
            output_size=dist_layer.get_total_param_size(),
            activation=activation,
        )

        super().__init__(dense_layers + [dist_layer])

        negative_log_likelihood = lambda y, p_y: -p_y.log_prob(y)
        self.compile(
            optimizer=tf.compat.v2.optimizers.Adam(0.01), loss=negative_log_likelihood
        )

    def score(self, x_test, y_test):
        assert x_test.shape[0] == y_test.shape[0]

        output = self(x_test)
        return tf.reduce_sum(output.log_prob(y_test), axis=0)

    @staticmethod
    def _get_dense_layers(hidden_sizes, output_size, activation="relu"):
        hidden = [
            tf.keras.layers.Dense(size, activation=activation) for size in hidden_sizes
        ]
        output = tf.keras.layers.Dense(output_size, activation="linear")
        return hidden + [output]
