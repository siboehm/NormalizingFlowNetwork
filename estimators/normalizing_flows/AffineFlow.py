import tensorflow_probability as tfp


class AffineFlow(tfp.bijectors.Affine):
    def __init__(self, t, n_dims, name="AffineFlow"):
        assert t.shape[-1] == 2 * n_dims
        super(AffineFlow, self).__init__(
            shift=t[..., 0:n_dims], scale_diag=1.0 + t[..., n_dims : 2 * n_dims], name=name
        )

    @staticmethod
    def get_param_size(n_dims):
        """
        :param n_dims:  The dimension of the distribution to be transformed by the flow
        :return: (int) The dimension of the parameter space for the flow
        """
        return 2 * n_dims
