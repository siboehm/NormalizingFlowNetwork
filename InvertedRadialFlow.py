import tensorflow as tf
import tensorflow_probability as tfp


class InvertedRadialFlow(tfp.bijectors.Bijector):
    # implements x = y + (alpha * beta * (y - y_0)) / (alpha + |y - y0|)
    # This parametrization is different from the original one proposed in (Rezende, Mohammed)
    _alpha = None
    _beta = None
    _gamma = None
    _n_dims = None

    def __init__(
        self, alpha, beta, gamma, n_dims, validate_args=False, name="InvertedRadialFlow"
    ):
        """
        Parameter shapes (assuming you're transforming a distribution over d-space):
        shape alpha = (1, )
        shape beta = (1, )
        shape z0 = (1, )
        """
        super(InvertedRadialFlow, self).__init__(
            validate_args=validate_args, inverse_min_event_ndims=0, name=name
        )
        self._alpha = InvertedRadialFlow._handle_input_dimensionality(alpha)
        self._beta = InvertedRadialFlow._handle_input_dimensionality(beta)
        self._gamma = InvertedRadialFlow._handle_input_dimensionality(gamma)
        self._n_dims = n_dims

    @staticmethod
    def _handle_input_dimensionality(z):
        """
        If rank(z) is 1, increase rank to 2
        We want tensors of shape (?, 1)
        """
        return tf.cond(
            tf.equal(tf.rank(z), tf.rank([0.0])),
            lambda: tf.expand_dims(z, 1),
            lambda: z,
        )

    def _r(self, z):
        return tf.reduce_sum(tf.abs(z - self._gamma), 1, keepdims=True)

    def _h(self, r):
        return 1.0 / (self._alpha + r)

    def _inverse(self, z):
        """
        Runs a forward pass through the bijector
        """
        z = InvertedRadialFlow._handle_input_dimensionality(z)
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
        det = (1.0 + ab * h) ** (self._n_dims - 1) * (1.0 + ab * h + ab * der_h * r)
        return tf.log(det)
