import tensorflow as tf
import tensorflow_probability as tfp
import pytest
import numpy as np
from estimators import NormalizingFlowNetwork

tfd = tfp.distributions
tf.random.set_seed(22)
np.random.seed(22)


@pytest.mark.slow
def test_x_noise_reg():
    x_train = np.linspace(-3, 3, 300, dtype=np.float32).reshape((300, 1))
    noise = tfd.MultivariateNormalDiag(loc=5 * tf.math.sin(2 * x_train), scale_diag=abs(x_train))
    y_train = noise.sample().numpy()

    too_much_noise = NormalizingFlowNetwork(
        1,
        n_flows=2,
        hidden_sizes=(16, 16),
        noise_reg=("fixed_rate", 3.0),
        trainable_base_dist=True,
    )

    too_much_noise.fit(x_train, y_train, epochs=700, verbose=0)

    x_test = np.linspace(-3, 3, 300, dtype=np.float32).reshape((300, 1))
    noise = tfd.MultivariateNormalDiag(loc=5 * tf.math.sin(2 * x_test), scale_diag=abs(x_test))
    y_test = noise.sample().numpy()
    out1 = too_much_noise.pdf(x_test, y_test).numpy()
    out2 = too_much_noise.pdf(x_test, y_test).numpy()
    # making sure that the noise regularisation is deactivated in testing mode
    assert all(out1 == out2)

    little_noise = NormalizingFlowNetwork(
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

    noise = NormalizingFlowNetwork(
        3,
        n_flows=3,
        hidden_sizes=(16, 16),
        trainable_base_dist=True,
        noise_reg=("fixed_rate", 1.0),
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
