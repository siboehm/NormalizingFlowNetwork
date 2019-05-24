import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python import tf2
from plotting import plot_affine_transformed_dist, plot_dist
from NFDistributionLayer import NFDistributionLayer

if not tf2.enabled():
    import tensorflow.compat.v2 as tf

    tf.enable_v2_behavior()
    assert tf2.enabled()

tfd = tfp.distributions


class NF_ML_Estimator(tf.keras.Sequential):
    def __init__(
        self,
        n_dims,
        flow_types=("radial", "radial"),
        hidden_sizes=(16, 16),
        train_base_dist=True,
        activation="relu",
    ):
        dist_layer = NFDistributionLayer(
            flow_types=flow_types, n_dims=n_dims, trainable_base_dist=train_base_dist
        )
        dense_layers = self._get_dense_layers(
            hidden_sizes=hidden_sizes,
            output_size=dist_layer.get_total_param_size(),
            activation=activation,
        )

        super().__init__(dense_layers + [dist_layer])

    @staticmethod
    def _get_dense_layers(hidden_sizes, output_size, activation="relu"):
        hidden = [
            tf.keras.layers.Dense(size, activation=activation) for size in hidden_sizes
        ]
        output = tf.keras.layers.Dense(output_size, activation="linear")
        return hidden + [output]
