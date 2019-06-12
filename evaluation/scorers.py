import numpy as np

# this score exists in it's own module to allow sklearn to parallelize evaluation better
# this is a score, therefore higher is better


def bayesian_log_likelihood_score(wrapped_model, x, y, **kwargs):
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    posterior_draws = 50
    loss = 0
    nll = wrapped_model.model._get_neg_log_likelihood()
    for _ in range(posterior_draws):
        loss += nll(y, wrapped_model.model(x, training=False)).numpy().mean()
    return -loss / posterior_draws


def mle_log_likelihood_score(wrapped_model, x, y, **kwargs):
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    nll = wrapped_model.model._get_neg_log_likelihood()
    return -nll(y, wrapped_model.model(x, training=False)).numpy().mean()
