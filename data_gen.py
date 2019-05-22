import numpy as np


# a cosine + some noise
def gen_cosine_noise_data(num, noise_std=0.2, heterosced_noise=0.0):
    x = np.linspace(-4, 4, num=num)
    y_heterosced_noise = np.zeros(x.shape)
    for i in range(int(num / 2)):
        y_heterosced_noise[i] = abs(x[i])
    y = (
        4 * np.sin(x)
        + np.random.normal(0, noise_std, size=num)
        + y_heterosced_noise * np.random.normal(0, heterosced_noise, size=num)
    )
    x = x.astype(np.float32).reshape((num, 1))
    y = y.astype(np.float32).reshape((num, 1))
    assert x.shape == y.shape
    return x, y


def gen_trippe_hetero_data(
    dim=1, n_pts=10000, bimodal=False, heteroscedastic=True, asymetric=False
):
    """gen_synthetic_data generates synthetic data with 1D output which is
    bimodal and heteroscedastic.
    Args:
        dim: dimensionality of the input
        n_pts: number of points to generate
        bimodal: true to generate bimodal data
        heteroscedastic: true to generate heteroscedastic data
        asymetric: set True to have noise have asymetric tails
    """
    bounds = [-np.pi, np.pi]
    range_size = bounds[1] - bounds[0]
    if heteroscedastic:
        noise_scale = 3.0  # for noise
    else:
        noise_scale = 0.0  # for noise

    global_noise = 0.0  # additional homoscedastic gaussian noise
    signal_scale = 5.0

    if bimodal:
        n_pts_mode = int(n_pts / 2.0)
    else:
        n_pts_mode = n_pts

    # Generate X for first half of data
    X = np.random.uniform(bounds[0], bounds[1], size=[n_pts_mode, dim])
    Y_std = (
        noise_scale * abs(np.sin(X).prod(axis=1)) + global_noise
    )  # Heteroscedastic noise
    if asymetric:
        Y = abs(np.random.normal(0.0, abs(Y_std))) + signal_scale * np.sin(X).prod(
            axis=1
        )
    else:
        Y = np.random.normal(0.0, abs(Y_std)) + signal_scale * np.sin(X).prod(axis=1)

    # Generate data from second mode
    if bimodal:
        X_more = np.random.uniform(bounds[0], bounds[1], size=[n_pts_mode, dim])
        Y_std_more = noise_scale * abs(np.sin(X_more)).prod(axis=1) + global_noise
        # The bimodality arises from using 'abs(X_more)' rather than simply
        # X_more within sin
        if asymetric:
            Y_more = abs(
                np.random.normal(0.0, abs(Y_std_more))
            ) + signal_scale * np.sin(abs(X_more).prod(axis=1))
        else:
            Y_more = np.random.normal(0.0, abs(Y_std_more)) + signal_scale * np.sin(
                abs(X_more).prod(axis=1)
            )
        # concatenate two datasets together for bimodal signal
        X, Y = np.array(list(X) + list(X_more)), np.array(list(Y) + list(Y_more))

    Y = Y.reshape([n_pts, 1])
    return X, Y
