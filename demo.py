import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python import tf2

if not tf2.enabled():
    import tensorflow.compat.v2 as tf

    tf.enable_v2_behavior()
    assert tf2.enabled()

import numpy as np
import matplotlib.pyplot as plt
from data_gen import gen_trippe_hetero_data
from MaximumLikelihoodNFEstimator import MaximumLikelihoodNFEstimator
from plotting import plot_dist

tfd = tfp.distributions
tf.random.set_seed(22)

x_train, y_train = gen_trippe_hetero_data(
    1, n_pts=400, heteroscedastic=True, bimodal=True
)
# x_train, y_train = gen_cosine_noise_data(300, noise_std=0.3, heterosced_noise=0.5)

model = MaximumLikelihoodNFEstimator(
    n_dims=1,
    flow_types=("radial", "radial"),
    hidden_sizes=(16, 16),
    trainable_base_dist=True,
    activation="relu",
)

model.fit(x_train, y_train, epochs=1000, verbose=2)
x_swoop = np.linspace(-3, 3, num=100).reshape((100, 1))
result_dist = model(x_swoop)
plot_dist(x_swoop, dist=result_dist, y_range=[-12, 12], y_num=100)
plt.show()
