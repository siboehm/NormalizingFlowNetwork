import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from simulation.dummy_data_gen import gen_cosine_noise_data, gen_trippe_hetero_data
from estimators import *
from cde.density_simulation import GaussianMixture, SkewNormal
from evaluation.scorers import mle_log_likelihood_score, DummySklearWrapper
from evaluation.visualization.flow_plotting import plot_model

tfd = tfp.distributions
np.random.seed(22)

x_train, y_train = gen_cosine_noise_data(300, noise_std=0.3, heterosced_noise=0.5)
x_test, y_test = gen_cosine_noise_data(3000, noise_std=0.3, heterosced_noise=0.5)
plt.scatter(x_train, y_train)
plt.show()


model = BayesKernelMixtureNetwork(
    kl_weight_scale=0.3 / x_train.shape[0],
    hidden_sizes=(7, 7),
    n_dims=1,
    n_centers=20,
    activation="tanh",
    learning_rate=1e-2,
    map_mode=False,
)
model.fit(x_train, y_train, epochs=2000, verbose=2)
print(mle_log_likelihood_score(DummySklearWrapper(model), x_test, y_test))

x_swoop = np.linspace(-4, 4, num=100).reshape((100, 1))
for _ in range(1):
    plot_model(x_swoop, model, y_num=100, y_range=[-6, 6])
    plt.show()
