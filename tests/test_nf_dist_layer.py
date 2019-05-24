import tensorflow as tf
import pytest
import tensorflow_probability as tfp
from tensorflow.python import tf2
from NFDistributionLayer import NFDistributionLayer
from normalizing_flows import FLOWS

if not tf2.enabled():
    import tensorflow.compat.v2 as tf

    tf.enable_v2_behavior()
    assert tf2.enabled()
tfd = tfp.distributions


def test_total_param_size():
    layer1 = NFDistributionLayer(
        ("planar", "radial", "affine"), n_dims=1, trainable_base_dist=False
    )
    layer2 = NFDistributionLayer(
        ("planar", "radial", "affine"), n_dims=3, trainable_base_dist=True
    )
    assert layer1.get_total_param_size() == 3 + 3 + 2
    assert layer2.get_total_param_size() == (3 + 3 + 1) + (3 + 1 + 1) + (3 + 3) + (
        3 + 3
    )


def test_get_bijector():
    output1 = NFDistributionLayer._get_bijector(
        tf.zeros((10, 8)), ("planar", "radial", "affine"), 1
    )
    assert len(output1.bijectors) == 3
    assert output1.inverse_min_event_ndims == 1
    assert type(output1.bijectors[0]) == FLOWS["affine"]
    assert type(output1.bijectors[1]) == FLOWS["radial"]
    assert type(output1.bijectors[2]) == FLOWS["planar"]

    output1 = NFDistributionLayer._get_bijector(
        tf.zeros((10, 9)), ("planar", "radial"), 2
    )
    assert len(output1.bijectors) == 2
    assert output1.inverse_min_event_ndims == 1

    with pytest.raises(AssertionError):
        NFDistributionLayer._get_bijector(tf.zeros((10, 8)), ("planar", "radial"), 2)
