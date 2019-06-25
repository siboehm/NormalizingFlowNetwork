import tensorflow as tf
import pytest
import tensorflow_probability as tfp
from estimators.DistributionLayers import (
    InverseNormalizingFlowLayer,
    MeanFieldLayer,
    GaussianMixtureLayer,
    GaussianKernelsLayer,
)
from estimators.normalizing_flows import FLOWS
import numpy as np

tfd = tfp.distributions


def test_total_param_size_nf():
    layer1 = InverseNormalizingFlowLayer(
        ("planar", "radial", "affine"), n_dims=1, trainable_base_dist=False
    )
    layer2 = InverseNormalizingFlowLayer(
        ("planar", "radial", "affine"), n_dims=3, trainable_base_dist=True
    )
    assert layer1.get_total_param_size() == 3 + 3 + 2
    assert layer2.get_total_param_size() == (3 + 3 + 1) + (3 + 1 + 1) + (3 + 3) + (3 + 3)


def test_total_param_size_mf():
    layer1 = MeanFieldLayer(n_dims=10, scale=None)
    layer2 = MeanFieldLayer(n_dims=10, scale=10.0)
    assert layer1.get_total_param_size() == 20
    assert layer2.get_total_param_size() == 10


def test_total_param_size_mixture():
    layer1 = GaussianMixtureLayer(n_dims=5, n_centers=5)
    layer2 = GaussianMixtureLayer(n_dims=1, n_centers=3)
    assert layer1.get_total_param_size() == 55
    assert layer2.get_total_param_size() == 9


def test_mixture_dist_fn():
    dist_fn = GaussianMixtureLayer._get_distribution_fn(n_dims=1, n_centers=5)
    dist = dist_fn(tf.ones((1, 15)))
    assert dist.event_shape == [1]
    assert dist.batch_shape == [1]

    dist = dist_fn(tf.ones((3, 15)))
    assert dist.event_shape == [1]
    assert dist.batch_shape == [3]

    with pytest.raises(ValueError):
        dist_fn(tf.ones((10, 10)))

    dist_fn = GaussianMixtureLayer._get_distribution_fn(n_dims=3, n_centers=5)

    dist = dist_fn(tf.ones((1, 35)))
    assert dist.event_shape == [3]
    assert dist.batch_shape == [1]

    dist = dist_fn(tf.ones((3, 35)))
    assert dist.event_shape == [3]
    assert dist.batch_shape == [3]

    with pytest.raises(ValueError):
        dist_fn(tf.ones((10, 12)))


def test_mf_dist_fn_trainable_scale():
    dist_fn = MeanFieldLayer._get_distribution_fn(n_dims=10, scale=None)
    dist = dist_fn(tf.ones((20,)))
    assert dist.event_shape == [10]
    assert dist.batch_shape == []

    dist = dist_fn(tf.ones((10, 20)))
    assert dist.event_shape == [10]
    assert dist.batch_shape == [10]

    with pytest.raises(AssertionError):
        dist_fn(tf.ones((10, 19)))


def test_mf_dist_fn_fixed_scale():
    dist_fn = MeanFieldLayer._get_distribution_fn(n_dims=10, scale=10.0)

    dist = dist_fn(tf.ones((10,)))
    assert dist.event_shape == [10]
    assert dist.batch_shape == []

    dist = dist_fn(tf.ones((1, 10)))
    assert dist.event_shape == [10]
    assert dist.batch_shape == [1]

    dist = dist_fn(tf.ones((10, 10)))
    assert dist.event_shape == [10]
    assert dist.batch_shape == [10]

    with pytest.raises(AssertionError):
        dist_fn(tf.ones((10, 9)))


def test_gk_dist_fn():
    y_train = np.linspace(-1, 1, 100).reshape((100, 1))

    layer = GaussianKernelsLayer(
        n_centers=10, n_dims=1, trainable_scale=True, init_scales=(0.3, 0.7)
    )
    assert layer.get_total_param_size() == 22

    dist_fn = layer._get_distribution_fn()

    assert np.sum(np.abs(layer.locs.numpy())) == 0.0
    layer.set_center_points(y_train)
    assert not np.sum(np.abs(layer.locs.numpy())) == 0.0

    dist = dist_fn(tf.ones((1, 22)))
    assert dist.event_shape == [1]
    assert dist.batch_shape == [1]
    assert dist.sample().shape == (1, 1)

    dist = dist_fn(tf.ones((3, 22)))
    assert dist.event_shape == [1]
    assert dist.batch_shape == [3]
    assert dist.sample().shape == (3, 1)

    with pytest.raises(AssertionError):
        dist_fn(tf.ones((10, 19)))

    y_train = np.linspace(-1, 1, 200).reshape((100, 2))
    layer = GaussianKernelsLayer(
        n_centers=10, n_dims=2, trainable_scale=True, init_scales=(0.3, 0.7)
    )
    assert layer.get_total_param_size() == 22

    dist_fn = layer._get_distribution_fn()

    assert np.all(layer.locs[0].numpy() == [[0.0, 0.0]])
    layer.set_center_points(y_train)
    assert not np.any(layer.locs[0].numpy() == [[0.0, 0.0]])

    dist = dist_fn(tf.ones((1, 22)))
    assert dist.event_shape == [2]
    assert dist.batch_shape == [1]
    assert dist.sample().shape == (1, 2)

    dist = dist_fn(tf.ones((3, 22)))
    assert dist.event_shape == [2]
    assert dist.batch_shape == [3]
    assert dist.sample().shape == (3, 2)

    with pytest.raises(AssertionError):
        dist_fn(tf.ones((10, 19)))


def test_nf_dist_fn():
    dist_fn = InverseNormalizingFlowLayer._get_distribution_fn(
        n_dims=1, flow_types=("radial", "planar"), trainable_base_dist=False
    )
    dist = dist_fn(tf.ones((1, 6)))
    assert dist.event_shape == [1]
    assert dist.batch_shape == [1]

    dist = dist_fn(tf.ones((3, 6)))
    assert dist.event_shape == [1]
    assert dist.batch_shape == [3]

    with pytest.raises(AssertionError):
        dist_fn(tf.ones((10, 7)))

    dist_fn = InverseNormalizingFlowLayer._get_distribution_fn(
        n_dims=2, flow_types=("radial", "planar"), trainable_base_dist=True
    )

    dist = dist_fn(tf.ones((1, 13)))
    assert dist.event_shape == [2]
    assert dist.batch_shape == [1]

    dist = dist_fn(tf.ones((3, 13)))
    assert dist.event_shape == [2]
    assert dist.batch_shape == [3]

    with pytest.raises(AssertionError):
        dist_fn(tf.ones((10, 12)))


def test_get_bijector():
    output1 = InverseNormalizingFlowLayer._get_bijector(
        tf.zeros((10, 8)), ("planar", "radial", "affine"), 1
    )
    assert len(output1.bijectors) == 3
    assert output1.inverse_min_event_ndims == 1
    assert type(output1.bijectors[0]) == FLOWS["affine"]
    assert type(output1.bijectors[1]) == FLOWS["radial"]
    assert type(output1.bijectors[2]) == FLOWS["planar"]

    output1 = InverseNormalizingFlowLayer._get_bijector(tf.zeros((10, 9)), ("planar", "radial"), 2)
    assert len(output1.bijectors) == 2
    assert output1.inverse_min_event_ndims == 1

    with pytest.raises(AssertionError):
        InverseNormalizingFlowLayer._get_bijector(tf.zeros((10, 8)), ("planar", "radial"), 2)
