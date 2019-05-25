import tensorflow as tf
from tensorflow.python import tf2

if not tf2.enabled():
    import tensorflow.compat.v2 as tf

    tf.enable_v2_behavior()
    assert tf2.enabled()

import tensorflow_probability as tfp

tfd = tfp.distributions

import numpy as np
from MaximumLikelihoodNFEstimator import MaximumLikelihoodNFEstimator
from BayesianNFEstimator import BayesianNFEstimator


def test_dense_layer_generation():
    layers = MaximumLikelihoodNFEstimator._get_dense_layers((2, 2, 2), 2)
    assert len(layers) == 4


def test_on_gaussian():
    tf.random.set_random_seed(22)
    np.random.seed(22)
    # sinusoidal data with heteroscedastic noise
    x_train = np.linspace(-3, 3, 300, dtype=np.float32).reshape((300, 1))
    noise = tfd.MultivariateNormalDiag(
        loc=5 * tf.math.sin(2 * x_train), scale_diag=abs(x_train)
    )
    y_train = noise.sample().numpy()

    model = MaximumLikelihoodNFEstimator(
        1,
        flow_types=("radial", "radial", "radial"),
        hidden_sizes=(16, 16),
        trainable_base_dist=True,
    )
    model.fit(x_train, y_train, epochs=700, verbose=0)

    x_test = np.linspace(-3, 3, 1000, dtype=np.float32).reshape((1000, 1))
    noise = tfd.MultivariateNormalDiag(
        loc=5 * tf.math.sin(2 * x_test), scale_diag=abs(x_test)
    )
    y_test = noise.sample().numpy()

    output = model(x_test)
    score = (
        tf.reduce_sum(abs(output.prob(y_test) - noise.prob(y_test)), axis=0) / 1000.0
    )
    assert score < 0.45


def test_bimodal_gaussian():
    tf.random.set_random_seed(22)
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

    model = MaximumLikelihoodNFEstimator(
        1,
        flow_types=("radial", "radial"),
        hidden_sizes=(16, 16),
        trainable_base_dist=True,
    )

    model.fit(x_train, y_train, epochs=700, verbose=0)

    x_test, y_test, pdf = get_data(800)

    output = model(x_test)
    score = tf.reduce_sum(abs(output.prob(y_test) - pdf.prob(y_test)), axis=0) / 800.0
    assert score < 0.1
