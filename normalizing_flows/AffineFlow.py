import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python import tf2
if not tf2.enabled():
    import tensorflow.compat.v2 as tf
    tf.enable_v2_behavior()
    assert tf2.enabled()


class AffineFlow(tfp.bijectors.Bijector):

    def __init__(self, a, b, validate_args=False, name="InvertedRadialFlow"):
        """
        Parameter shapes (assuming you're transforming a distribution over d-space):
        shape alpha = (1, )
        shape beta = (1, )
        shape z0 = (1, )
        """
        super(AffineFlow, self).__init__(forward_min_event_ndims=0)
        assert len(a.shape) == len(b.shape) == 2
        assert a.shape[1] == b.shape[1]

        self._a = a
        self._b = b

    def _forward(self, x):
        """
        Forward pass through the bijector. a*x + b
        """
        return self._a * x + self._b

    def _inverse(self, y):
        """
        Backward pass through the bijector. (y-b) / a
        """
        return (y - self._b) / self._a

    def _inverse_log_det_jacobian(self, y):
        return -tf.reduce_sum(self._a, 1, keep_dims=True)
