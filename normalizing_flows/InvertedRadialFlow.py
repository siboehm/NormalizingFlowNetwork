import tensorflow as tf
import tensorflow_probability as tfp


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

    def __init__(self, a, b, g, validate_args=False, name='InvertedRadialFlow'):
        """
        Parameter shapes (assuming you're transforming a distribution over d-space):

        shape alpha = (?, 1)
        shape beta = (?, 1)
        shape gamma = (?, ndims)
        """
        super(InvertedRadialFlow, self).__init__(validate_args=validate_args, name=name, forward_min_event_ndims=0)

        # split the input parameter into the individual parameters alpha, beta, gamma

        # constraining the parameters before they are assigned to ensure invertibility
        print(a, b, g)
        self._alpha = a
        self._beta = b
        self._gamma = g

    @staticmethod
    def get_param_size(n_dims):
        """
        :param n_dims:  The dimension of the distribution to be transformed by the flow
        :return: (int) The dimension of the parameter space for the flow
        """
        return 1 + 1 + n_dims

    def _r(self, z):
        return tf.reduce_sum(tf.abs(z - self._gamma), 1, keepdims=True)

    def _h(self, r):
        return 1. / (self._alpha + r)

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
        h = self._h(r)
        der_h = tf.gradients(h, [r])[0]
        ab = self._alpha * self._beta
        det = (1. + ab * h) ** (self.n_dims - 1) * (1. + ab * h + ab * der_h * r)
        return tf.log(det)

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
        return tf.exp(beta) - 1.
