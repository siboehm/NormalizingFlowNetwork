import numpy as np
import scipy

# this score exists in it's own module to allow sklearn to parallelize evaluation better
# this is a score, therefore higher is better


class DummySklearWrapper:
    def __init__(self, model):
        self.model = model


def bayesian_log_likelihood_score(wrapped_model, x, y, **kwargs):
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    scores = None
    nll = wrapped_model.model._get_neg_log_likelihood()
    posterior_draws = 1 if wrapped_model.model.map_mode else 50
    for _ in range(posterior_draws):
        res = np.expand_dims(-nll(y, wrapped_model.model.call(x, training=False)).numpy(), axis=0)
        scores = res if scores is None else np.concatenate([scores, res], axis=0)
        print(scores.shape)
    logsumexp = scipy.special.logsumexp(scores, axis=0) - np.log(posterior_draws)
    print(logsumexp.shape)
    return logsumexp.mean()


def mle_log_likelihood_score(wrapped_model, x, y, **kwargs):
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    nll = wrapped_model.model._get_neg_log_likelihood()
    return -nll(y, wrapped_model.model(x, training=False)).numpy().mean()
