import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

import pytest
import numpy as np
from estimators import MaximumLikelihoodNFEstimator


tf.random.set_random_seed(22)
np.random.seed(22)


def test_dense_layer_generation():
    layers = MaximumLikelihoodNFEstimator(1)._get_dense_layers(
        hidden_sizes=(2, 2, 2), output_size=2, activation="linear"
    )
    assert len(layers) == 6


def test_model_output_dims_1d():
    x_train = np.linspace(-1, 1, 10).reshape((10, 1))
    y_train = np.linspace(-1, 1, 10).reshape((10, 1))

    m1 = MaximumLikelihoodNFEstimator(
        1, n_flows=3, hidden_sizes=(16, 16), trainable_base_dist=False
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

    m1 = MaximumLikelihoodNFEstimator(1, n_flows=0, hidden_sizes=(16, 16), trainable_base_dist=True)
    m1.fit(x_train, y_train, epochs=1, verbose=0)
    output = m1(x_train)
    assert isinstance(output, tfd.TransformedDistribution)
    assert output.event_shape == [1]
    assert output.batch_shape == [10]
    assert output.log_prob([[0.0]]).shape == [10]


def test_model_ouput_dims_3d():
    x_train = np.linspace([[-1]] * 3, [[1]] * 3, 10).reshape((10, 3))
    y_train = np.linspace([[-1]] * 3, [[1]] * 3, 10).reshape((10, 3))

    m1 = MaximumLikelihoodNFEstimator(3, n_flows=3, hidden_sizes=(16, 16), trainable_base_dist=True)
    m1.fit(x_train, y_train, epochs=1, verbose=0)
    output = m1(x_train)
    assert isinstance(output, tfd.TransformedDistribution)
    assert output.event_shape == [3]
    assert output.batch_shape == [10]
    assert output.log_prob([[0.0] * 3]).shape == [10]


@pytest.mark.slow
def test_x_noise_reg():
    x_train = np.linspace(-3, 3, 300, dtype=np.float32).reshape((300, 1))
    noise = tfd.MultivariateNormalDiag(loc=5 * tf.math.sin(2 * x_train), scale_diag=abs(x_train))
    y_train = noise.sample().numpy()

    too_much_noise = MaximumLikelihoodNFEstimator(
        1, n_flows=2, hidden_sizes=(16, 16), noise_reg=("fixed_rate", 3.0), trainable_base_dist=True
    )

    too_much_noise.fit(x_train, y_train, epochs=700, verbose=0)

    x_test = np.linspace(-3, 3, 300, dtype=np.float32).reshape((300, 1))
    noise = tfd.MultivariateNormalDiag(loc=5 * tf.math.sin(2 * x_test), scale_diag=abs(x_test))
    y_test = noise.sample().numpy()
    out1 = too_much_noise.pdf(x_test, y_test).numpy()
    out2 = too_much_noise.pdf(x_test, y_test).numpy()
    # making sure that the noise regularisation is deactivated in testing mode
    assert all(out1 == out2)

    little_noise = MaximumLikelihoodNFEstimator(
        1,
        n_flows=2,
        hidden_sizes=(16, 16),
        noise_reg=("rule_of_thumb", 0.1),
        trainable_base_dist=True,
    )
    little_noise.fit(x_train, y_train, epochs=700, verbose=0)

    little_noise_score = tf.reduce_sum(little_noise.pdf(x_test, y_test)) / 700.0
    too_much_noise_score = tf.reduce_sum(too_much_noise.pdf(x_test, y_test)) / 700.0
    assert little_noise_score > too_much_noise_score


def test_y_noise_reg():
    x_train = np.linspace([[-1]] * 3, [[1]] * 3, 10, dtype=np.float32).reshape((10, 3))
    y_train = np.linspace([[-1]] * 3, [[1]] * 3, 10, dtype=np.float32).reshape((10, 3))

    noise = MaximumLikelihoodNFEstimator(
        3, n_flows=3, hidden_sizes=(16, 16), trainable_base_dist=True, noise_reg=("fixed_rate", 1.0)
    )
    noise.fit(x_train, y_train, epochs=10, verbose=0)

    input_model = noise._get_input_model()
    # y_input should not include randomness during evaluation
    y1 = input_model(y_train, training=False).numpy()
    y2 = input_model(y_train, training=False).numpy()
    assert np.all(y1 == y2)

    # loss should include randomness during learning
    y1 = input_model(y_train, training=True).numpy()
    y2 = input_model(y_train, training=True).numpy()
    assert not np.all(y1 == y2)


@pytest.mark.slow
def test_on_gaussian():
    tf.random.set_random_seed(22)
    np.random.seed(22)
    # sinusoidal data with heteroscedastic noise
    x_train = np.linspace(-3, 3, 300, dtype=np.float32).reshape((300, 1))
    noise = tfd.MultivariateNormalDiag(loc=5 * tf.math.sin(2 * x_train), scale_diag=abs(x_train))
    y_train = noise.sample().numpy()

    model = MaximumLikelihoodNFEstimator(
        1, n_flows=2, hidden_sizes=(16, 16), trainable_base_dist=True
    )
    model.fit(x_train, y_train, epochs=800, verbose=0)

    x_test = np.linspace(-3, 3, 1000, dtype=np.float32).reshape((1000, 1))
    noise = tfd.MultivariateNormalDiag(loc=5 * tf.math.sin(2 * x_test), scale_diag=abs(x_test))
    y_test = noise.sample().numpy()

    score = tf.reduce_sum(abs(model.pdf(x_test, y_test) - noise.prob(y_test)), axis=0) / 1000.0
    assert score < 0.45


@pytest.mark.slow
def test_bimodal_gaussian():
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
        1, n_flows=3, hidden_sizes=(10, 10), trainable_base_dist=True
    )

    model.fit(x_train, y_train, epochs=700, verbose=0)

    x_test, y_test, pdf = get_data(800)

    score = tf.reduce_sum(abs(model.pdf(x_test, y_test) - pdf.prob(y_test)), axis=0) / 800.0
    assert score < 0.1005
