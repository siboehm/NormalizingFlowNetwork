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
from normalizing_flows.InvertedRadialFlow import InvertedRadialFlow
from data_gen import gen_cosine_noise_data, gen_trippe_hetero_data
from normalizing_flows.AffineFlow import AffineFlow
from normalizing_flows.InvertedPlanarFlow import InvertedPlanarFlow
from normalizing_flows.InvertedRadialFlow import InvertedRadialFlow

x_train, y_train = gen_trippe_hetero_data(
    1, n_pts=400, heteroscedastic=True, bimodal=True
)
x_train, y_train = gen_cosine_noise_data(100, noise_std=0.3, heterosced_noise=0.5)
plt.scatter(x_train, y_train)
plt.show()


param_size = 3


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
            param_size,
            posterior_mean_field,
            prior,
            kl_weight=1 / x_train.shape[0],
            activation="linear",
        ),
        tfp.layers.DistributionLambda(make_distribution_fn=nf,
                                      convert_to_tensor_fn=lambda dist: dist.log_prob(0.)),
    ]
)

nf = lambda t: tfd.TransformedDistribution(
    distribution=tfd.MultivariateNormalDiag(
        loc=[[0.0]], scale_identity_multiplier=[[1.0]]
    ),
    bijector=tfp.bijectors.Chain(
        [
            InvertedRadialFlow(t[..., 2:3], t[..., 3:4], t[..., 4:5]),
            tfp.bijectors.Affine(shift=t[..., 0:1], scale_diag=t[..., 1:2]),
        ]
    ),
)


model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(16, activation="tanh"),
        tf.keras.layers.Dense(16, activation="tanh"),
        tf.keras.layers.Dense(5, activation="linear"),
        tfp.layers.DistributionLambda(
            make_distribution_fn=nf, convert_to_tensor_fn=lambda d: d.log_prob([1.0])
        ),
    ]
)

negative_log_likelihood = lambda y, p_y: -p_y.log_prob(y)
model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.01), loss=negative_log_likelihood
)

model.fit(x_train, y_train, epochs=1000, verbose=2)
x_test = np.linspace(-4, 4, num=100).reshape((100, 1))
result_dist = model(x_test)
print(result_dist)
plot_dist(x_test, dist=result_dist, y_range=[-8, 8], y_num=100)
plt.show()
