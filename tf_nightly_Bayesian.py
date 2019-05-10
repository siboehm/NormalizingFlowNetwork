import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python import tf2
from plotting import plot_affine_transformed_dist

if not tf2.enabled():
    import tensorflow.compat.v2 as tf

    tf.enable_v2_behavior()
    assert tf2.enabled()

tfd = tfp.distributions
from normalizing_flows.InvertedRadialFlow import InvertedRadialFlow
from data_gen import gen_cosine_noise_data, gen_trippe_hetero_data
from normalizing_flows.AffineFlow import AffineFlow

# x_train, y_train = gen_cosine_noise_data(100, noise_std=0.3, heterosced_noise=0.5)
x_train, y_train = gen_trippe_hetero_data(1, 200)


def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.0))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        tfp.layers.DistributionLambda(
            lambda t: tfd.Independent(
                tfd.Normal(loc=t[..., :n], scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
                reinterpreted_batch_ndims=1,
            )
        ),
    ]
    )


def prior(kernel_size, bias_size=0, dtype=None, trainable=False):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype, trainable=trainable),
        tfp.layers.DistributionLambda(
            lambda t: tfd.Independent(tfd.Normal(loc=t, scale=1.), reinterpreted_batch_ndims=1)
        ),
    ]
    )


model = tf.keras.Sequential([
    tfp.layers.DenseVariational(
        10,
        posterior_mean_field,
        prior,
        kl_weight=1 / x_train.shape[0],
        activation="relu",
    ),
    tfp.layers.DenseVariational(
        2,
        posterior_mean_field,
        prior,
        kl_weight=1 / x_train.shape[0],
        activation="linear",
    ),
    tfp.layers.DistributionLambda(
        lambda t: tfd.TransformedDistribution(
            distribution=tfd.Normal(loc=tf.zeros_like(t[..., 0:1]), scale=tf.ones_like(t[..., 0:1])),
            bijector=tfp.bijectors.AffineScalar(shift=t[..., 0:1], scale=tf.nn.softplus(0.05 * t[..., 1:2])),
        )
    ),
])

negative_log_likelihood = lambda y, p_y: -p_y.log_prob(y)
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.03), loss=negative_log_likelihood)
model.fit(x_train, y_train, epochs=1500, verbose=2)

x_test = np.linspace(-4, 4, num=500).reshape((500, 1))
y_preds = [model(x_test) for _ in range(30)]

plt.scatter(x_train, y_train, marker="+", label="Training data")
plot_affine_transformed_dist(x_test, y_preds[0], labels=True)
for y_pred in y_preds[1:]:
    plot_affine_transformed_dist(x_test, y_pred, labels=False)
plt.legend()
plt.show()
