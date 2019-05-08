import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python import tf2
from InvertedRadialFlow import InvertedRadialFlow

if not tf2.enabled():
    import tensorflow.compat.v2 as tf

    tf.enable_v2_behavior()
    assert tf2.enabled()

tfd = tfp.distributions

# a cosine + some noise
def build_toy_dataset(num, noise_std=0.2, heterosced_noise=0.0):
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


def gen_synthetic_data(
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

    global_noise = 1.0  # additional homoscedastic gaussian noise
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


x_train, y_train = gen_synthetic_data(
    dim=1, n_pts=200, bimodal=True, heteroscedastic=True, asymetric=False
)


def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.0))
    return tf.keras.Sequential(
        [
            tfp.layers.VariableLayer(2 * n, dtype=dtype),
            tfp.layers.DistributionLambda(
                lambda t: tfd.Independent(
                    tfd.Normal(
                        loc=t[..., :n], scale=1e-5 + tf.nn.softplus(c + t[..., n:])
                    ),
                    reinterpreted_batch_ndims=1,
                )
            ),
        ]
    )


def prior_trainable(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential(
        [
            tfp.layers.VariableLayer(n, dtype=dtype, trainable=True),
            tfp.layers.DistributionLambda(
                lambda t: tfd.Independent(
                    tfd.Normal(loc=t, scale=1), reinterpreted_batch_ndims=1
                )
            ),
        ]
    )


model = tf.keras.Sequential(
    [
        tfp.layers.DenseVariational(
            10,
            posterior_mean_field,
            prior_trainable,
            kl_weight=1 / x_train.shape[0],
            activation="relu",
        ),
        tfp.layers.DenseVariational(
            2,
            posterior_mean_field,
            prior_trainable,
            kl_weight=1 / x_train.shape[0],
            activation="linear",
        ),
        tfp.layers.DistributionLambda(
            lambda t: tfd.Normal(
                loc=t[..., :1], scale=1e-3 + tf.math.softplus(0.01 * t[..., 1:])
            )
        ),
    ]
)

nll = lambda y, p_y: -p_y.log_prob(y)
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), loss=nll)
model.fit(x_train, y_train, epochs=4500, verbose=2)

x_test = np.linspace(-5, 5, num=500).reshape((500, 1))
y_preds = [model(x_test) for _ in range(30)]


def plot_dist(x, dist, labels=True):
    assert isinstance(dist, tfp.distributions.Distribution)
    plt.plot(
        x_test, dist.mean(), "r", lw=2, alpha=0.1, label=("Mean" if labels else None)
    )
    plt.plot(
        x,
        dist.mean() + 2 * dist.stddev(),
        "g",
        lw=2,
        alpha=0.1,
        label=("mean + stddev" if labels else None),
    )
    plt.plot(
        x,
        dist.mean() - 2 * dist.stddev(),
        "g",
        lw=2,
        alpha=0.1,
        label=("mean - stddev" if labels else None),
    )


plt.scatter(x_train, y_train, marker="+", label="Training data")
plot_dist(x_test, y_preds[0], labels=True)
for y_pred in y_preds[1:]:
    plot_dist(x_test, y_pred, labels=False)
plt.title("Prediction")
plt.legend()
plt.show()
