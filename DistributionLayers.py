import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python import tf2
from normalizing_flows import FLOWS

if not tf2.enabled():
    import tensorflow.compat.v2 as tf

    tf.enable_v2_behavior()
    assert tf2.enabled()
tfd = tfp.distributions


class MeanFieldLayer(tfp.layers.DistributionLambda):
    n_dims, uniform_scale = None, None

    def __init__(self, n_dims, uniform_scale=False, dtype=None):
        """
        A subclass of Distribution Lambda. A layer that uses it's input to parametrize n_dims-many indepentent normal
        distributions (aka mean field)
        :param n_dims: Dimension of the distribution that's being output by the Layer
        """
        self.n_dims = n_dims
        self.uniform_scale = n_dims
        make_dist_fn = self._get_distribution_fn(n_dims, uniform_scale)
        super().__init__(make_distribution_fn=make_dist_fn, dtype=dtype)

    @staticmethod
    def _get_distribution_fn(n_dims, uniform_scale):
        c = tf.math.log(tf.math.expm1(1.0))
        scale = (
            lambda t: 1.0
            if uniform_scale
            else 1e-5 + tf.nn.softplus(c + t[..., n_dims:])
        )

        def make_dist_fn(t):
            assert t.shape[-1] == (n_dims if uniform_scale else 2 * n_dims)
            tfd.Independent(
                tfd.Normal(loc=t[..., :n_dims], scale=scale(t)),
                reinterpreted_batch_ndims=1,
            )

        return make_dist_fn

    def get_total_param_size(self):
        return self.n_dims if self.uniform_scale else 2 * self.n_dims


class InverseNormalizingFlowLayer(tfp.layers.DistributionLambda):
    _flow_types = None
    _trainable_base_dist = None
    _n_dims = None

    def __init__(self, flow_types, n_dims, trainable_base_dist=False):
        """
        Subclass of a DistributionLambda. A layer that uses it's input to parametrize a normalizing flow
        that transforms a base normal distribution
        :param flow_types: Types of flows to use, applied in order from base_dist -> transformed_dist
        :param n_dims: dimension of the underlying distribution being transformed
        :param trainable_base_dist: whether the base normal distribution should have trainable loc and scale diag
        """
        assert all([flow_type in FLOWS for flow_type in flow_types])

        self._flow_types = flow_types
        self._trainable_base_dist = trainable_base_dist
        self._n_dims = n_dims

        # as keras transforms tensors, this layer needs to have an tensor-like output
        # therefore a function needs to be provided that transforms a distribution into a tensor
        # per default the .sample() function is used, but our reversed flows cannot perform that operation
        convert_ttfn = lambda d: d.log_prob([1.0] * n_dims)
        make_flow_dist = lambda t: tfd.TransformedDistribution(
            distribution=self._get_base_dist(
                (t[..., 0 : 2 * n_dims] if trainable_base_dist else None), n_dims
            ),
            bijector=self._get_bijector(
                (t[..., 2 * n_dims :] if trainable_base_dist else t), flow_types, n_dims
            ),
        )
        super().__init__(
            make_distribution_fn=make_flow_dist, convert_to_tensor_fn=convert_ttfn
        )

    def get_total_param_size(self):
        """
        :return: The total number of parameters to specify this distribution
        """
        num_flow_params = sum(
            [
                FLOWS[flow_type].get_param_size(self._n_dims)
                for flow_type in self._flow_types
            ]
        )
        base_dist_params = 2 * self._n_dims if self._trainable_base_dist else 0
        return num_flow_params + base_dist_params

    @staticmethod
    def _get_bijector(t, flow_types, n_dims):
        # intuitively, we want to flows to go from base_dist -> transformed dist
        flow_types = list(reversed(flow_types))
        param_sizes = [
            FLOWS[flow_type].get_param_size(n_dims) for flow_type in flow_types
        ]
        assert sum(param_sizes) == t.shape[-1]
        split_beginnings = [sum(param_sizes[0:i]) for i in range(len(param_sizes))]
        chain = [
            FLOWS[flow_type](t[..., begin : begin + size], n_dims)
            for begin, size, flow_type in zip(split_beginnings, param_sizes, flow_types)
        ]
        return tfp.bijectors.Chain(chain)

    @staticmethod
    def _get_base_dist(t, n_dims):
        if t is None:
            return tfd.MultivariateNormalDiag(
                loc=[[0.0] * n_dims], scale_diag=[[1.0] * n_dims]
            )
        else:
            assert t.shape[-1] == 2 * n_dims
            return tfd.MultivariateNormalDiag(
                loc=t[..., 0:n_dims],
                scale_diag=tf.math.softplus(0.05 * t[..., n_dims : 2 * n_dims]),
            )
