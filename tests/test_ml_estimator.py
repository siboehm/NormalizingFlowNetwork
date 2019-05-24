import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python import tf2
from NF_ML_Estimator import NF_ML_Estimator

if not tf2.enabled():
    import tensorflow.compat.v2 as tf

    tf.enable_v2_behavior()
    assert tf2.enabled()
tfd = tfp.distributions


def test_dense_layer_generation():
    layers = NF_ML_Estimator._get_dense_layers((2, 2, 2), 2)
    assert len(layers) == 4
