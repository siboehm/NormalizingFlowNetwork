import tensorflow as tf
from tensorflow.python import tf2
import pytest

if not tf2.enabled():
    import tensorflow.compat.v2 as tf

    tf.enable_v2_behavior()
    assert tf2.enabled()

import tensorflow_probability as tfp

tfd = tfp.distributions

import numpy as np
from BayesianNFEstimator import BayesianNFEstimator

tf.random.set_seed(22)
np.random.seed(22)


def test_dense_layer_generation():
    layers = BayesianNFEstimator._get_dense_layers((2, 2, 2), 2, None, None, 1)
    assert len(layers) == 4


def test_model_output_dims_1d():
    x_train = np.linspace(-1, 1, 10).reshape((10, 1))
    y_train = np.linspace(-1, 1, 10).reshape((10, 1))

    m1 = BayesianNFEstimator(
        1,
        kl_norm_const=x_train.shape[0],
        flow_types=("radial", "affine", "planar"),
        hidden_sizes=(16, 16),
        trainable_base_dist=False,
    )
    m1.fit(x_train, y_train, epochs=1, verbose=0)
    output = m1(x_train)
    assert isinstance(output, tfd.TransformedDistribution)
    assert output.event_shape == [1]
    assert output.batch_shape == [10]
    assert output.log_prob([[0.0]]).shape == [10]


def test_model_output_dims_1d_2():
    x_train = np.linspace(-1, 1, 10).reshape((10, 1))
    y_train = np.linspace(-1, 1, 10).reshape((10, 1))

    m1 = BayesianNFEstimator(
        1,
        kl_norm_const=x_train.shape[0],
        flow_types=tuple(),
        hidden_sizes=(16, 16),
        trainable_base_dist=True,
    )
    m1.fit(x_train, y_train, epochs=1, verbose=0)
    output = m1(x_train)
    assert isinstance(output, tfd.TransformedDistribution)
    assert output.event_shape == [1]
    assert output.batch_shape == [10]
    assert output.log_prob([[0.0]]).shape == [10]


def test_model_ouput_dims_3d():
    x_train = np.linspace([[-1]] * 3, [[1]] * 3, 10).reshape((10, 3))
    y_train = np.linspace([[-1]] * 3, [[1]] * 3, 10).reshape((10, 3))

    m1 = BayesianNFEstimator(
        3,
        kl_norm_const=x_train.shape[0],
        flow_types=("radial", "affine", "planar"),
        hidden_sizes=(16, 16),
        trainable_base_dist=True,
    )
    m1.fit(x_train, y_train, epochs=1, verbose=0)
    output = m1(x_train)
    assert isinstance(output, tfd.TransformedDistribution)
    assert output.event_shape == [3]
    assert output.batch_shape == [10]
    assert output.log_prob([[0.0] * 3]).shape == [10]


@pytest.mark.slow
def test_bayesian_nn_on_gaussian():
    # sinusoidal data with heteroscedastic noise
    x_train = np.linspace(-3, 3, 300, dtype=np.float32).reshape((300, 1))
    noise = tfd.MultivariateNormalDiag(
        loc=5 * tf.math.sin(2 * x_train), scale_diag=abs(x_train)
    )
    y_train = noise.sample().numpy()

    model = BayesianNFEstimator(
        1,
        flow_types=tuple(),
        kl_norm_const=x_train.shape[0],
        hidden_sizes=(10,),
        activation="tanh",
        learning_rate=0.03,
        trainable_base_dist=True,
    )
    model.fit(x_train, y_train, epochs=1000, verbose=0)

    x_test = np.linspace(-3, 3, 1000, dtype=np.float32).reshape((1000, 1))
    noise = tfd.MultivariateNormalDiag(
        loc=5 * tf.math.sin(2 * x_test), scale_diag=abs(x_test)
    )
    y_test = noise.sample().numpy()

    score = 0
    for _ in range(30):
        output = model(x_test)
        draw = (
            tf.reduce_sum(abs(output.prob(y_test) - noise.prob(y_test)), axis=0)
            / 1000.0
        )
        score += draw
    assert score / 30.0 < 0.62


@pytest.mark.slow
def test_bimodal_gaussian():
    tf.random.set_seed(22)
    np.random.seed(22)

    def get_data(sample_size=400):
        noise = tfd.Mixture(
            cat=tfd.Categorical(probs=[0.5, 0.5]),
            components=[
                tfd.MultivariateNormalDiag(loc=[3.0], scale_diag=[0.5]),
                tfd.MultivariateNormalDiag(loc=[-3.0], scale_diag=[0.5]),
            ],
        )
        x = np.linspace(-3, 3, sample_size, dtype=np.float32).reshape((sample_size, 1))
        y = noise.sample(sample_size).numpy()
        return x, y, noise

    x_train, y_train, _ = get_data()

    model = BayesianNFEstimator(
        1,
        flow_types=("radial",),
        learning_rate=0.02,
        hidden_sizes=(10,),
        kl_norm_const=x_train.shape[0],
        trainable_base_dist=True,
        activation="tanh",
    )

    model.fit(x_train, y_train, epochs=2000, verbose=0)

    x_test, y_test, pdf = get_data(1000)

    score = 0
    for _ in range(30):
        output = model(x_test)
        draw = (
            tf.reduce_sum(abs(output.prob(y_test) - pdf.prob(y_test)), axis=0) / 1000.0
        )
        score += draw
    assert score / 30.0 < 0.18
