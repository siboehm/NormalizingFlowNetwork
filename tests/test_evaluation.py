import numpy as np
from estimators import BayesianNFEstimator, MaximumLikelihoodNFEstimator
from evaluation.scorers import bayesian_log_likelihood_score, mle_log_likelihood_score


class DummyWrapper:
    def __init__(self, model):
        self.model = model


def test_bayesian_score():
    x_train = np.linspace(-1, 1, 10).reshape((10, 1))
    y_train = np.linspace(-1, 1, 10).reshape((10, 1))

    mle = MaximumLikelihoodNFEstimator(
        1, flow_types=tuple(), hidden_sizes=(6, 6), trainable_base_dist=True
    )
    mle.fit(x_train, y_train, epochs=10, verbose=0)
    # deterministic, so should be the same
    assert bayesian_log_likelihood_score(
        DummyWrapper(mle), x_train, y_train
    ) == -mle.evaluate(x_train, y_train)

    be = BayesianNFEstimator(
        n_dims=1, flow_types=tuple(), hidden_sizes=(6, 6), trainable_base_dist=True
    )
    be.fit(x_train, y_train, epochs=10, verbose=0)
    # random, so shouldn't be the same
    assert bayesian_log_likelihood_score(
        DummyWrapper(be), x_train, y_train
    ) != -be.evaluate(x_train, y_train)


def test_mle_score():
    x_train = np.linspace(-1, 1, 10).reshape((10, 1))
    y_train = np.linspace(-1, 1, 10).reshape((10, 1))

    mle = MaximumLikelihoodNFEstimator(
        1, flow_types=tuple(), hidden_sizes=(6, 6), trainable_base_dist=True
    )
    mle.fit(x_train, y_train, epochs=10, verbose=0)
    # deterministic, so should be the same
    assert mle_log_likelihood_score(
        DummyWrapper(mle), x_train, y_train
    ) == -mle.evaluate(x_train, y_train)
