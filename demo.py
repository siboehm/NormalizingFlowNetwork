import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python import tf2

if not tf2.enabled():
    import tensorflow.compat.v2 as tf

    tf.enable_v2_behavior()
    assert tf2.enabled()

import numpy as np
import matplotlib.pyplot as plt
from data_gen import gen_trippe_hetero_data, gen_cosine_noise_data
from estimators import BayesianNFEstimator
from estimators import MaximumLikelihoodNFEstimator

from plotting import plot_model

tfd = tfp.distributions
tf.random.set_seed(22)
np.random.seed(22)


# x_train, y_train = gen_trippe_hetero_data(
#   1, n_pts=400, heteroscedastic=True, bimodal=True
# )
x_train, y_train = gen_cosine_noise_data(300, noise_std=0.3, heterosced_noise=0.5)
plt.scatter(x_train, y_train)
plt.show()


model = BayesianNFEstimator(
    n_dims=1,
    kl_weight_scale=1./x_train.shape[0],
    flow_types=("radial",),
    hidden_sizes=(10,),
    trainable_base_dist=True,
    activation="tanh",
    trainable_prior=False,
)
# ml_model = MaximumLikelihoodNFEstimator(
#     n_dims=1,
#     flow_types=("radial", "radial"),
#     hidden_sizes=(16, 16),
#     trainable_base_dist=False,
# )

model.fit(x_train, y_train, epochs=1000, verbose=2)

x_swoop = np.linspace(-4, 4, num=100).reshape((100, 1))
for _ in range(4):
    plot_model(x_swoop, model, y_num=100, y_range=[-6, 6])
    plt.show()
