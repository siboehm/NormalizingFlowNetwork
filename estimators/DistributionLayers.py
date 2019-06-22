import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python import tf2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances
from estimators.normalizing_flows import FLOWS

if not tf2.enabled():
    import tensorflow.compat.v2 as tf

    tf.enable_v2_behavior()
    assert tf2.enabled()
tfd = tfp.distributions


class MeanFieldLayer(tfp.layers.DistributionLambda):
    def __init__(self, n_dims, scale=None, map_mode=False, dtype=None):
        """
        A subclass of Distribution Lambda. A layer that uses it's input to parametrize n_dims-many indepentent normal
        distributions (aka mean field)
        Requires input size n_dims for fixed scale, 2*n_dims for trainable scale
        Mean Field also works for scalars
        The input tensors for this layer should be initialized to Zero for a standard normal distribution
        :param n_dims: Dimension of the distribution that's being output by the Layer
        :param scale: (float) None if scale should be trainable. If not None, specifies the fixed scale of the
            independent normals. If map mode is activated, this is ignored and set to 1.0
        """
        self.n_dims = n_dims
        self.scale = scale

        if map_mode:
            self.scale = 1.0
        convert_ttf = tfd.Distribution.mean if map_mode else tfd.Distribution.sample

        make_dist_fn = self._get_distribution_fn(self.n_dims, self.scale)

        super().__init__(
            make_distribution_fn=make_dist_fn, convert_to_tensor_fn=convert_ttf, dtype=dtype
        )

    @staticmethod
    def _get_distribution_fn(n_dims, scale=None):
        if scale is None:

            def dist_fn(t):
                assert t.shape[-1] == 2 * n_dims
                return tfd.Independent(
                    tfd.Normal(
                        loc=t[..., 0:n_dims],
                        scale=1e-3
                        + tf.nn.softplus(
                            tf.math.log(tf.math.expm1(1.0)) + 0.05 * t[..., n_dims : 2 * n_dims]
                        ),
                    ),
                    reinterpreted_batch_ndims=1,
                )

        else:
            assert scale > 0.0

            def dist_fn(t):
                assert t.shape[-1] == n_dims
                return tfd.Independent(
                    tfd.Normal(loc=t[..., 0:n_dims], scale=scale), reinterpreted_batch_ndims=1
                )

        return dist_fn

    def get_total_param_size(self):
        return 2 * self.n_dims if self.scale is None else self.n_dims


class GaussianKernelsLayer(tfp.layers.DistributionLambda):
    def __init__(self, n_centers, n_dims, trainable_scale=True, init_scales=(0.3, 0.7)):
        self.n_centers = n_centers
        self.n_scales = len(init_scales)
        self.n_dims = n_dims
        self.trainable_scale = trainable_scale

        self.locs = [
            tf.Variable(initial_value=np.zeros((1, n_dims)), dtype=tf.float32, trainable=False)
            for _ in range(n_centers)
        ]

        self.scales = [
            tf.Variable(initial_value=scale, dtype=tf.float32, trainable=trainable_scale)
            for scale in tf.math.log(tf.math.expm1(init_scales))
        ]
        assert len(self.scales) == len(init_scales)
        super().__init__(make_distribution_fn=self._get_distribution_fn())

    def get_total_param_size(self):
        """
        :return: The total number of parameters to specify this distribution
        """
        return self.n_centers * self.n_scales

    def _get_distribution_fn(self):
        def dist(t):
            assert t.shape[-1] == self.n_centers * self.n_scales
            return tfd.Mixture(
                components=[
                    tfd.MultivariateNormalDiag(
                        loc=loc * tf.ones_like(t[..., 0 : self.n_dims]),
                        scale_identity_multiplier=tf.nn.softplus(scale),
                    )
                    for loc in self.locs
                    for scale in self.scales
                ],
                cat=tfd.Categorical(logits=t[..., 0 : self.n_centers * self.n_scales]),
            )

        return dist

    def set_center_points(self, y):
        ndim_y = y.shape[1]
        n_edge_points = min(2 * ndim_y, self.n_centers // 2)

        # select 2*n_edge_points that are the farthest away from mean
        fathest_points_idx = np.argsort(np.linalg.norm(y - y.mean(axis=0), axis=1))[
            -2 * n_edge_points :
        ]
        Y_farthest = y[np.ix_(fathest_points_idx)]

        # choose points among Y farthest so that pairwise cosine similarity maximized
        dists = cosine_distances(Y_farthest)
        selected_indices = [0]
        for _ in range(1, n_edge_points):
            idx_greatest_distance = np.argsort(
                np.min(dists[np.ix_(range(Y_farthest.shape[0]), selected_indices)], axis=1), axis=0
            )[-1]
            selected_indices.append(idx_greatest_distance)
        centers_at_edges = Y_farthest[np.ix_(selected_indices)]

        # remove selected centers from Y
        indices_to_remove = fathest_points_idx[np.ix_(selected_indices)]
        y = np.delete(y, indices_to_remove, axis=0)

        # adjust k such that the final output has size k
        k = self.n_centers - n_edge_points

        model = KMeans(n_clusters=k, n_jobs=-2)
        model.fit(y)
        cluster_centers = model.cluster_centers_
        cluster_centers = np.concatenate([centers_at_edges, cluster_centers], axis=0)

        for loc, val in zip(self.locs, cluster_centers):
            loc.assign(tf.expand_dims(np.float32(val), axis=0))


class GaussianMixtureLayer(tfp.layers.DistributionLambda):
    def __init__(self, n_centers, n_dims):
        """
        Subclass of DistributionLambda. A layer that uses it's input to parametrize a mixture of gaussians with
        variable scales, locs and mixture params.
        :param n_centers: The amount of guassian to mix
        :param n_dims: The dimension of the resulting distribution
        """
        self._n_centers = n_centers
        self._n_dims = n_dims
        make_mixture_dist = self._get_distribution_fn(n_centers, n_dims)
        super().__init__(make_distribution_fn=make_mixture_dist)

    def get_total_param_size(self):
        """
        :return: The total number of parameters to specify this mixture distribution
        """
        mixture_params = self._n_centers
        gaussian_params = 2 * self._n_dims * self._n_centers
        return mixture_params + gaussian_params

    @staticmethod
    def _get_distribution_fn(n_centers, n_dims):
        return lambda t: tfd.Mixture(
            components=[
                tfd.MultivariateNormalDiag(
                    loc=t[..., loc_start : loc_start + n_dims],
                    scale_diag=tf.nn.softplus(
                        0.05 * t[..., loc_start + n_dims : loc_start + 2 * n_dims]
                        + tf.math.log(tf.math.expm1(1.0))
                    ),
                )
                for loc_start in range(0, 2 * n_centers * n_dims, 2 * n_dims)
            ],
            cat=tfd.Categorical(
                probs=tf.nn.softmax(
                    t[..., 2 * n_centers * n_dims : 2 * n_centers * n_dims + n_centers]
                )
            ),
        )


class InverseNormalizingFlowLayer(tfp.layers.DistributionLambda):
    _flow_types = None
    _trainable_base_dist = None
    _n_dims = None

    def __init__(self, flow_types, n_dims, trainable_base_dist=False):
        """
        Subclass of a DistributionLambda. A layer that uses it's input to parametrize a normalizing flow
        that transforms a base normal distribution
        This layer does not work for scalars!
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
        convert_ttfn = lambda d: tf.zeros(n_dims)
        make_flow_dist = self._get_distribution_fn(n_dims, flow_types, trainable_base_dist)
        super().__init__(make_distribution_fn=make_flow_dist, convert_to_tensor_fn=convert_ttfn)

    @staticmethod
    def _get_distribution_fn(n_dims, flow_types, trainable_base_dist):
        return lambda t: tfd.TransformedDistribution(
            distribution=InverseNormalizingFlowLayer._get_base_dist(t, n_dims, trainable_base_dist),
            bijector=InverseNormalizingFlowLayer._get_bijector(
                (t[..., 2 * n_dims :] if trainable_base_dist else t), flow_types, n_dims
            ),
        )

    def get_total_param_size(self):
        """
        :return: The total number of parameters to specify this distribution
        """
        num_flow_params = sum(
            [FLOWS[flow_type].get_param_size(self._n_dims) for flow_type in self._flow_types]
        )
        base_dist_params = 2 * self._n_dims if self._trainable_base_dist else 0
        return num_flow_params + base_dist_params

    @staticmethod
    def _get_bijector(t, flow_types, n_dims):
        # intuitively, we want to flows to go from base_dist -> transformed dist
        flow_types = list(reversed(flow_types))
        param_sizes = [FLOWS[flow_type].get_param_size(n_dims) for flow_type in flow_types]
        assert sum(param_sizes) == t.shape[-1]
        split_beginnings = [sum(param_sizes[0:i]) for i in range(len(param_sizes))]
        chain = [
            FLOWS[flow_type](t[..., begin : begin + size], n_dims)
            for begin, size, flow_type in zip(split_beginnings, param_sizes, flow_types)
        ]
        return tfp.bijectors.Chain(chain)

    @staticmethod
    def _get_base_dist(t, n_dims, trainable):
        if trainable:
            return tfd.MultivariateNormalDiag(
                loc=t[..., 0:n_dims],
                scale_diag=1e-3
                + tf.math.softplus(
                    tf.math.log(tf.math.expm1(1.0)) + 0.05 * t[..., n_dims : 2 * n_dims]
                ),
            )
        else:
            # we still need to know the batch size, therefore we need t for reference
            return tfd.MultivariateNormalDiag(loc=tf.zeros_like(t[..., 0:n_dims]))
