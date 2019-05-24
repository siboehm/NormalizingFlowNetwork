import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python import tf2
from plotting import plot_affine_transformed_dist, plot_dist

if not tf2.enabled():
    import tensorflow.compat.v2 as tf

    tf.enable_v2_behavior()
    assert tf2.enabled()

tfd = tfp.distributions
from data_gen import gen_cosine_noise_data, gen_trippe_hetero_data
from NFDistributionLayer import NFDistributionLayer

tf.random.set_seed(22)
x_train, y_train = gen_trippe_hetero_data(
    1, n_pts=400, heteroscedastic=True, bimodal=True
)
# x_train, y_train = gen_cosine_noise_data(300, noise_std=0.3, heterosced_noise=0.5)
plt.scatter(x_train, y_train)
plt.show()


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


def prior(kernel_size, bias_size=0, dtype=None, trainable=False):
    n = kernel_size + bias_size
    return tf.keras.Sequential(
        [
            tfp.layers.VariableLayer(n, dtype=dtype, trainable=trainable),
            tfp.layers.DistributionLambda(
                lambda t: tfd.Independent(
                    tfd.Normal(loc=t, scale=1.0), reinterpreted_batch_ndims=1
                )
            ),
        ]
    )


dist_layer = NFDistributionLayer(('radial', 'radial'), 1, True)
model = tf.keras.Sequential(
    [
        tfp.layers.DenseVariational(
            10,
            posterior_mean_field,
            prior,
            kl_weight=1 / x_train.shape[0],
            activation="relu",
        ),
        tfp.layers.DenseVariational(
            dist_layer.get_total_param_size(),
            posterior_mean_field,
            prior,
            kl_weight=1 / x_train.shape[0],
            activation="linear",
        ),
        dist_layer
    ]
)


model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(dist_layer.get_total_param_size(), activation="linear"),
        dist_layer
    ]
)

negative_log_likelihood = lambda y, p_y: -p_y.log_prob(y)
model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.01), loss=negative_log_likelihood
)

model.fit(x_train, y_train, epochs=1000, verbose=2)
x_test = np.linspace(-3, 3, num=100).reshape((100, 1))
result_dist = model(x_test)
plot_dist(x_test, dist=result_dist, y_range=[-12, 12], y_num=100)
plt.show()
