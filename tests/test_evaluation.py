import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from estimators import BayesianNFEstimator, NormalizingFlowEstimator
from evaluation.scorers import bayesian_log_likelihood_score, mle_log_likelihood_score
import pytest

tf.random.set_random_seed(22)


class DummyWrapper:
    def __init__(self, model):
        self.model = model


def test_bayesian_score():
    # sinusoidal data with heteroscedastic noise
    x_train = np.linspace(-3, 3, 300, dtype=np.float32).reshape((300, 1))
    noise = tfp.distributions.MultivariateNormalDiag(
        loc=5 * tf.math.sin(2 * x_train), scale_diag=abs(x_train)
    )
    y_train = noise.sample().numpy()

    mle = NormalizingFlowEstimator(1, n_flows=0, hidden_sizes=(6, 6), trainable_base_dist=True)
    mle.fit(x_train, y_train, epochs=20, verbose=0)
    # deterministic, so should be the same
    # mle furthermore has no regularisation loss / KL-divs added, therefore evaluate and nll are the same
    assert bayesian_log_likelihood_score(DummyWrapper(mle), x_train, y_train) == pytest.approx(
        -mle.evaluate(x_train, y_train)
    )

    be = BayesianNFEstimator(
        n_dims=1,
        kl_weight_scale=1.0 / x_train.shape[0],
        n_flows=0,
        hidden_sizes=(6, 6),
        trainable_base_dist=True,
    )
    be.fit(x_train, y_train, epochs=200, verbose=0)
    score = bayesian_log_likelihood_score(DummyWrapper(be), x_train, y_train)
    loss = sum([be.evaluate(x_train, y_train) for _ in range(50)]) / 50
    # as the loss has the KL div to the prior added to it, it's negative has to be smaller than the nll score
    assert score > -loss


def test_mle_score():
    x_train = np.linspace(-1, 1, 10).reshape((10, 1))
    y_train = np.linspace(-1, 1, 10).reshape((10, 1))

    mle = NormalizingFlowEstimator(1, n_flows=0, hidden_sizes=(6, 6), trainable_base_dist=True)
    mle.fit(x_train, y_train, epochs=10, verbose=0)
    # deterministic, so should be the same
    assert mle_log_likelihood_score(DummyWrapper(mle), x_train, y_train) == pytest.approx(
        -mle.evaluate(x_train, y_train)
    )
