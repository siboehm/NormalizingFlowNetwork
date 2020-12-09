import tensorflow as tf
import tensorflow_probability as tfp
import pytest
import numpy as np
from estimators import NormalizingFlowNetwork, MixtureDensityNetwork, KernelMixtureNetwork

tfd = tfp.distributions
tf.random.set_seed(22)
np.random.seed(22)


def test_dense_layer_generation():
    layers = NormalizingFlowNetwork(1)._get_dense_layers(
        hidden_sizes=(2, 2, 2), output_size=2, activation="linear"
    )
    assert len(layers) == 6


def model_output_dims_1d_testing(model):
    x_train = np.linspace(-1, 1, 10).reshape((10, 1))
    y_train = np.linspace(-1, 1, 10).reshape((10, 1))

    model.fit(x_train, y_train, epochs=1, verbose=0)
    output = model(x_train)
    assert isinstance(output, tfd.Distribution)
    assert output.event_shape == [1]
    assert output.batch_shape == [10]
    assert output.log_prob([[0.0]]).shape == [10]


def model_output_dims_3d_testing(model):
    x_train = np.linspace([[-1]] * 3, [[1]] * 3, 10).reshape((10, 3))
    y_train = np.linspace([[-1]] * 3, [[1]] * 3, 10).reshape((10, 3))

    model.fit(x_train, y_train, epochs=1, verbose=0)
    output = model(x_train)
    assert isinstance(output, tfd.Distribution)
    assert output.event_shape == [3]
    assert output.batch_shape == [10]
    assert output.log_prob([[0.0] * 3]).shape == [10]


def on_sinusoidal_gaussian_testing(model, expected_score):
    tf.random.set_seed(22)
    np.random.seed(22)

    # sinusoidal data with heteroscedastic noise
    def get_data(sample_size=400):
        x = np.linspace(-3, 3, sample_size, dtype=np.float32).reshape((sample_size, 1))
        noise = tfd.MultivariateNormalDiag(loc=5 * tf.math.sin(2 * x), scale_diag=abs(x))
        y = noise.sample().numpy()
        return x, y, noise

    x_train, y_train, _ = get_data(400)
    model.fit(x_train, y_train, epochs=800, verbose=0)

    x_test, y_test, pdf = get_data(1000)
    score = tf.reduce_sum(abs(model.pdf(x_test, y_test) - pdf.prob(y_test)), axis=0) / 1000.0
    assert score < expected_score


def on_bimodal_gaussian_testing(model, expected_score):
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

    x_train, y_train, _ = get_data(400)
    model.fit(x_train, y_train, epochs=800, verbose=0)

    x_test, y_test, pdf = get_data(1000)
    score = tf.reduce_sum(abs(model.pdf(x_test, y_test) - pdf.prob(y_test)), axis=0) / 1000.0

    assert score < expected_score


def test_ml_dims():
    # test 1 D
    for model in [
        NormalizingFlowNetwork(1, n_flows=1, hidden_sizes=(2, 2), trainable_base_dist=False),
        MixtureDensityNetwork(1, n_centers=2, hidden_sizes=(2, 2)),
        KernelMixtureNetwork(1, n_centers=3, hidden_sizes=(2, 2)),
    ]:
        model_output_dims_1d_testing(model)

    # test 3 D
    for model in [
        NormalizingFlowNetwork(3, n_flows=1, hidden_sizes=(2, 2), trainable_base_dist=True),
        MixtureDensityNetwork(3, n_centers=2, hidden_sizes=(2, 2)),
        KernelMixtureNetwork(3, n_centers=3, hidden_sizes=(2, 2)),
    ]:
        model_output_dims_3d_testing(model)


@pytest.mark.slow
def test_ml_nf_fitting():
    m1 = NormalizingFlowNetwork(1, n_flows=3, hidden_sizes=(10, 10), trainable_base_dist=True)
    on_sinusoidal_gaussian_testing(m1, 0.45)
    m2 = NormalizingFlowNetwork(1, n_flows=3, hidden_sizes=(10, 10), trainable_base_dist=True)
    on_bimodal_gaussian_testing(m2, 0.1012)


@pytest.mark.slow
def test_ml_mdn_fitting():
    m1 = MixtureDensityNetwork(1, n_centers=5, hidden_sizes=(10, 10))
    on_sinusoidal_gaussian_testing(m1, 0.46)
    m2 = MixtureDensityNetwork(1, n_centers=5, hidden_sizes=(10, 10))
    on_bimodal_gaussian_testing(m2, 0.1012)


@pytest.mark.slow
def test_ml_kmn_fitting():
    m1 = KernelMixtureNetwork(1, n_centers=25, hidden_sizes=(10, 10))
    on_sinusoidal_gaussian_testing(m1, 0.565)
    m2 = MixtureDensityNetwork(1, n_centers=25, hidden_sizes=(10, 10))
    on_bimodal_gaussian_testing(m2, 0.1058)
