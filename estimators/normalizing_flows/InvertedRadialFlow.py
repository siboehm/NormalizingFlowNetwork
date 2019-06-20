import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python import tf2

if not tf2.enabled():
    import tensorflow.compat.v2 as tf

    tf.enable_v2_behavior()
    assert tf2.enabled()


class InvertedRadialFlow(tfp.bijectors.Bijector):
    """
    Implements a bijector x = y + (alpha * beta * (y - y_0)) / (alpha + abs(y - y_0)).
    Args:
        params: Tensor shape (?, n_dims+2). This will be split into the parameters
            alpha (?, 1), beta (?, 1), gamma (?, n_dims).
            Furthermore alpha will be constrained to assure the invertability of the flow
        n_dims: The dimension of the distribution that will be transformed
        name: The name to give this particular flow
    """

    _alpha = None
    _beta = None
    _gamma = None

    def __init__(self, t, n_dims, name="InvertedRadialFlow"):
        super(InvertedRadialFlow, self).__init__(
            validate_args=False, name=name, inverse_min_event_ndims=1
        )

        assert t.shape[-1] == n_dims + 2
        alpha = t[..., 0:1]
        beta = t[..., 1:2]
        gamma = t[..., 2 : n_dims + 2]

        # constraining the parameters before they are assigned to ensure invertibility.
        # slightly shift alpha, softplus(zero centered input - 2) = small
        self._alpha = self._alpha_circ(0.3 * alpha - 2.0)
        # slightly shift beta, softplus(zero centered input + ln(e - 1)) = 0
        self._beta = self._beta_circ(0.1 * beta + tf.math.log(tf.math.expm1(1.0)))
        self._gamma = gamma

    @staticmethod
    def get_param_size(n_dims):
        """
        :param n_dims:  The dimension of the distribution to be transformed by the flow
        :return: (int) The dimension of the parameter space for the flow
        """
        return 1 + 1 + n_dims

    def _r(self, z):
        return tf.math.reduce_sum(tf.abs(z - self._gamma), 1, keepdims=True)

    def _h(self, r):
        return 1.0 / (self._alpha + r)

    def _inverse(self, z):
        """
        Runs a forward pass through the bijector
        """
        r = self._r(z)
        h = self._h(r)
        return z + (self._alpha * self._beta * h) * (z - self._gamma)

    def _inverse_log_det_jacobian(self, z):
        """
        Computes the ln of the absolute determinant of the jacobian
        """
        r = self._r(z)
        with tf.GradientTape() as g:
            g.watch(r)
            h = self._h(r)
        der_h = g.gradient(h, r)
        ab = self._alpha * self._beta
        det = (1.0 + ab * h) ** (1 - 1) * (1.0 + ab * h + ab * der_h * r)
        det = tf.squeeze(det, axis=-1)
        return tf.math.log(det)

    @staticmethod
    def _alpha_circ(alpha):
        """
        Method for constraining the alpha parameter to meet the invertibility requirements
        """
        return tf.nn.softplus(alpha)

    @staticmethod
    def _beta_circ(beta):
        """
        Method for constraining the beta parameter to meet the invertibility requirements
        """
        return tf.nn.softplus(beta) - 1.0
