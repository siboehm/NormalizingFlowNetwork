import matplotlib.pyplot as plt
import tensorflow_probability as tfp


def plot_affine_transformed_dist(x, dist, labels=True):
    assert isinstance(dist, tfp.distributions.Distribution)
    plt.plot(x, dist.mean(), "r", lw=2, alpha=0.1, label=("Mean" if labels else None))
    plt.plot(x, dist.mean() + dist.bijector.scale, "g", lw=2, alpha=0.1,
             label=("mean + affine scale" if labels else None))
    plt.plot(x, dist.mean() - dist.bijector.scale, "g", lw=2, alpha=0.1,
             label=("mean - affine scale" if labels else None))
