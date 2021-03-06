import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import numpy as np


def plot_affine_transformed_dist(x, dist, labels=True):
    assert isinstance(dist, tfp.distributions.Distribution)
    plt.plot(x, dist.mean(), "r", lw=2, alpha=0.1, label=("Mean" if labels else None))


def plot_dist(dist, x_range, y_range, num=100, labels=True):
    assert isinstance(dist, tfp.distributions.Distribution)
    assert len(y_range) == 2 and y_range[0] < y_range[1]
    assert len(x_range) == 2 and y_range[0] < y_range[1]

    # plots go from top to bottom for the columns
    x = np.linspace(x_range[1], x_range[0], num=num)
    y = np.linspace(y_range[1], y_range[0], num=num)

    dist_heatmap = np.zeros((num, num))
    for y_i in range(num - 1, -1, -1):
        for x_i in range(num):
            dist_heatmap[x_i, y_i] = dist.prob([x[x_i], y[y_i]])

    assert dist_heatmap.shape == (num, len(x))
    plt.imshow(dist_heatmap, aspect="equal")
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("y0")
    plt.ylabel("y1")


def plot_model(x, model, y_range, y_num=100):
    dist = model(x)
    assert isinstance(dist, tfp.distributions.Distribution)
    assert len(x.shape) == 2 and x[0][0] < x[-1][0]
    assert len(y_range) == 2 and y_range[0] < y_range[1]

    # plots go from top to bottom for the columns
    y_orig = np.linspace(y_range[1], y_range[0], num=y_num).reshape((y_num, 1))
    y = (y_orig - model.y_mean) / model.y_std

    dist_heatmap = np.zeros((y_num, len(x)))
    for i in range(y_num):
        dist_heatmap[i] = dist.prob(y[i]) / np.sum(model.y_std)

    assert dist_heatmap.shape == (y_num, len(x))
    plt.imshow(dist_heatmap, aspect="equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xticks([0, x.shape[0] - 1], [x[0][0], x[-1][0]])
    plt.yticks([0, y.shape[0] - 1], [y_orig[0][0], y_orig[-1][0]])
    plt.colorbar()
