import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


class InvertedPlanarFlow(tfp.bijectors.Bijector):
    """
    Implements a bijector x = y + u * tanh(w_t * y + b)

    Args:
        params: Tensor shape (?, 2*n_dims+1). This will be split into the parameters
            u (?, n_dims), w (?, n_dims), b (?, 1).
            Furthermore u will be constrained to assure the invertability of the flow
        n_dims: The dimension of the distribution that will be transformed
        name: The name to give this particular flow

    """

    _u, _w, _b = None, None, None

    def __init__(self, t, n_dims, name="Inverted_Planar_Flow"):
        super(InvertedPlanarFlow, self).__init__(
            validate_args=False,
            name=name,
            inverse_min_event_ndims=1,
        )
        assert t.shape[-1] == 2 * n_dims + 1
        u, w, b = t[..., 0:n_dims], t[..., n_dims:2*n_dims], t[..., 2*n_dims: 2*n_dims + 1]

        # constrain u before assigning it
        self._u = self._u_circ(u, w)
        self._w = u
        self._b = b

    @staticmethod
    def get_param_size(n_dims):
        """
        :param n_dims: The dimension of the distribution to be transformed by the flow
        :return: (int array) The dimension of the parameter space for this flow, n_dims + n_dims + 1
        """
        return n_dims + n_dims + 1

    @staticmethod
    def _u_circ(u, w):
        """
        To ensure invertibility of the flow, the following condition needs to hold: w_t * u >= -1
        :return: The transformed u
        """
        wtu = tf.math.reduce_sum(w * u, 1, keepdims=True)
        # add constant to make it more numerically stable
        m_wtu = -1.0 + tf.nn.softplus(wtu) + 1e-3
        norm_w_squared = tf.math.reduce_sum(w ** 2, 1, keepdims=True)
        return u + (m_wtu - wtu) * (w / norm_w_squared)

    def _wzb(self, z):
        """
        Computes w_t * z + b
        """
        return tf.math.reduce_sum(self._w * z, 1, keepdims=True) + self._b

    @staticmethod
    def _der_tanh(z):
        """
        Computes the derivative of hyperbolic tangent
        """
        return 1.0 - tf.math.tanh(z) ** 2

    def _inverse(self, z):
        """
        Runs a backward pass through the bijector
        @todo add the invertibility check again
        """
        return z + self._u * tf.math.tanh(self._wzb(z))

    def _inverse_log_det_jacobian(self, z):
        """
        Computes the ln of the absolute determinant of the jacobian
        """
        psi = self._der_tanh(self._wzb(z)) * self._w
        det_grad = 1.0 + tf.math.reduce_sum(self._u * psi, 1)
        return tf.math.log(tf.math.abs(det_grad))
