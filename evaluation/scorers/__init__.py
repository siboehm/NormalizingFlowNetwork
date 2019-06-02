# this score exists in it's own module to allow sklearn to parallelize evaluation better
# this is a score, therefore higher is better


def bayesian_log_likelihood_score(wrapped_model, x, y, **kwargs):
    posterior_draws = 30
    loss = 0
    for _ in range(posterior_draws):
        loss += wrapped_model.model.evaluate(x, y, verbose=0, **kwargs)
    return -loss / posterior_draws


def mle_log_likelihood_score(wrapped_model, x, y, **kwargs):
    return -wrapped_model.model.evaluate(x, y, verbose=0, **kwargs)
