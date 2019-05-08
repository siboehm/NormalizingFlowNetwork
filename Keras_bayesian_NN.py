import numpy as np
import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from keras import backend as K
from keras import activations, initializers, callbacks, optimizers
from keras.layers import Layer, Input, Dense, Activation
from keras.models import Sequential

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


def build_input_pipeline(x, y, batch_size):
    training_dataset = tf.data.Dataset.from_tensor_slices((x, y))
    training_batches = training_dataset.repeat().batch(batch_size)
    training_iterator = tf.data.make_one_shot_iterator(training_batches)
    batch_features, batch_labels = training_iterator.get_next()
    return batch_features, batch_labels


tf.set_random_seed(22)


n_data_points = 90
noise = 0.2
heteroscedastic_noise = 0.5
n_training_steps = 3500
model_heteroscedasticity = True
bayesian_nn = True
batch_size = 30
num_batches = n_data_points / batch_size
learning_rate = 0.05
activation_function = 'relu'

x_train, y_train = build_toy_dataset(n_data_points, noise, heteroscedastic_noise)


# defines the prior over weights as a standard Normal
weights_prior = tf.distributions.Normal(loc=0., scale=1.)


class DenseVariational(Layer):
    # instanciated the layer
    def __init__(self, output_dim, kl_loss_weight, **kwargs):
        self.output_dim = output_dim
        self.kl_loss_weight = kl_loss_weight
        super().__init__(**kwargs)

    def build(self, input_shape):

        # weight and bias are initialized as the pior
        self.kernel_mu = self.add_weight(
            name="kernel_mu",
            shape=(input_shape[1], self.output_dim),
            initializer="random_normal",
            trainable=True,
        )
        self.bias_mu = self.add_weight(
            name="bias_mu",
            shape=(self.output_dim,),
            initializer="random_normal",
            trainable=True,
        )
        self.kernel_rho = self.add_weight(
            name="kernel_rho",
            shape=(input_shape[1], self.output_dim),
            initializer="random_normal",
            trainable=True,
        )
        self.bias_rho = self.add_weight(
            name="bias_rho",
            shape=(self.output_dim,),
            initializer="random_normal",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        # constrain to make positive
        kernel_sigma = tf.math.softplus(self.kernel_rho)
        # the reparametrization trick
        kernel = self.kernel_mu + kernel_sigma * tf.random.normal(self.kernel_mu.shape)

        bias_sigma = tf.math.softplus(self.bias_rho)
        bias = self.bias_mu + bias_sigma * tf.random.normal(self.bias_mu.shape)

        # distance between variational family and prior is added as a loss
        self.add_loss(
            self.kl_loss(kernel, self.kernel_mu, kernel_sigma)
            + self.kl_loss(bias, self.bias_mu, bias_sigma)
        )

        # return the actual output of the layer
        return K.dot(x, kernel) + bias

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def kl_loss(self, w, mu, sigma):
        variational_dist = tf.distributions.Normal(mu, sigma)
        return kl_loss_weight * K.sum(
            variational_dist.log_prob(w) - weights_prior.log_prob(w)
        )


kl_loss_weight = 1.0 / num_batches

output_size = 2 if model_heteroscedasticity else 1
if bayesian_nn:
    model = Sequential(
        [
            DenseVariational(20, kl_loss_weight=kl_loss_weight, input_shape=(1,)),
            Activation(activation_function),
            DenseVariational(output_size, kl_loss_weight=kl_loss_weight),
        ]
    )
else:
    model = Sequential(
        [Dense(20, input_shape=(1,)), Activation(activation_function), Dense(output_size)]
    )


def neg_log_likelihood(y_true, y_pred):
    if model_heteroscedasticity:
        loc, rho = tf.split(y_pred, [1, 1], axis=1)
        dist = tf.distributions.Normal(loc=loc, scale=1e-3 + tf.nn.softplus(0.05 * rho))
    else:
        dist = tf.distributions.Normal(loc=y_pred, scale=1.0)
    return K.sum(-dist.log_prob(y_true))


model.compile(loss=neg_log_likelihood, optimizer=optimizers.Adam(lr=learning_rate))
model.fit(x_train, y_train, batch_size=batch_size, epochs=n_training_steps, verbose=2)


X_test = np.linspace(-6, 6, 1000).reshape((1000, 1))
y_pred_list = [model.predict(X_test) for _ in range(40)]


if model_heteroscedasticity:
    plt.plot(
        X_test, y_pred_list[0][:, 0], "r", lw=2, alpha=0.1, label="Posterior draws loc"
    )
    for y_pred in y_pred_list[1:]:
        plt.plot(X_test, y_pred[:, 0], "r", lw=2, alpha=0.07)
    plt.plot(
        X_test,
        y_pred_list[0][:, 1],
        "b",
        lw=2,
        alpha=0.1,
        label="Posterior draws scale",
    )
    for y_pred in y_pred_list[1:]:
        plt.plot(X_test, y_pred[:, 1], "b", lw=2, alpha=0.07)
else:
    plt.plot(X_test, y_pred_list[0], "r", lw=2, alpha=0.1, label="Posterior draws")
    for y_pred in y_pred_list[1:]:
        plt.plot(X_test, y_pred, "r", lw=2, alpha=0.07)

plt.scatter(x_train, y_train, marker="+", label="Training data")
plt.title("Prediction")
plt.legend()
plt.show()
#
# features, labels = build_input_pipeline(x_train, y_train, 32)
#
# model = tf.keras.Sequential(
#     [tfp.layers.DenseFlipout(10, activation=tf.nn.relu), tfp.layers.DenseFlipout(1)]
# )
# out = model(features)
# labels_distribution = tfd.Normal(loc=out, scale=1.0)
#
# neg_log_likelihood = -tf.reduce_mean(
#     input_tensor=labels_distribution.log_prob(labels)
# )
# kl = sum(model.losses) / n_data_points
# elbo_loss = neg_log_likelihood + kl
#
# optimizer = tf.train.AdamOptimizer()
# train_op = optimizer.minimize(elbo_loss)
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     sess.run(tf.local_variables_initializer())
#     for step in range(1000 + 1):
#         sess.run(train_op)
#         if step % 100 == 0:
#             loss_value = sess.run(elbo_loss)
#             print(loss_value)
#     print('I have finished')
