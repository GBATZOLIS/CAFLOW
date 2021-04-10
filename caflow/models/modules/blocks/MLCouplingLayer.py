from typing import Callable, Union
import torch
import torch.nn as nn
from FrEIA.modules import InvertibleModule 
from caflow.utils import logdist as logistic

class MLCouplingLayer(InvertibleModule):
    """Mixture-of-Logistics Coupling layer in Flow++

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the transformation network.
        num_blocks (int): Number of residual blocks in the transformation network.
        num_components (int): Number of components in the mixture.
        drop_prob (float): Dropout probability.
        use_attn (bool): Use attention in the NN blocks.
        aux_channels (int): Number of channels in optional auxiliary input.
    """
    def __init__(self, dims_in, dims_c=[], subnet_constructor: Callable = None, nn_settings=None):
        super().__init__(dims_in, dims_c)
        self.channels = dims_in[0][0]
        # ndims means the rank of tensor strictly speaking.
        # i.e. 1D, 2D, 3D tensor, etc.
        self.ndims = len(dims_in[0])
        self.split_len1 = self.channels // 2
        self.split_len2 = self.channels - self.channels // 2
        assert all([tuple(dims_c[i][1:]) == tuple(dims_in[0][1:]) for i in range(len(dims_c))]), \
            "Dimensions of input and one or more conditions don't agree."
        self.conditional = (len(dims_c) > 0)
        self.condition_length = sum([dims_c[i][0] for i in range(len(dims_c))])

        self.nn_type = nn_settings['nn_type']
        assert self.nn_type == 'nnflowpp', 'MixLog Coupling can only be used with the nnflowpp network for the time being.'
        coupling = 'MixLog'
        in_channels = self.split_len1
        aux_channels = self.condition_length if self.conditional else None
        out_channels = None
        num_channels_factor = nn_settings['num_channels_factor']
        num_blocks = nn_settings['num_blocks']
        drop_prob = nn_settings['drop_prob']
        use_attn = nn_settings['use_attn']
        num_components =  nn_settings['num_components']
        self.subnet = subnet_constructor(coupling=coupling, 
                        in_channels=in_channels, out_channels=out_channels, \
                        num_channels_factor=num_channels_factor, num_blocks=num_blocks, \
                        drop_prob=drop_prob, use_attn=use_attn, \
                        aux_channels=aux_channels, num_components=num_components)

    def forward(self, x, c=[], rev=False, jac=True):
        x_id, x_change = torch.split(x[0], [self.split_len1, self.split_len2], dim=1)
        aux = torch.cat([*c], dim=1) if self.conditional else None
        a, b, pi, mu, s = self.subnet(x_id, aux)

        if rev:
            out = x_change * a.mul(-1).exp() - b
            out, scale_ldj = logistic.inverse(out, reverse=True)
            out = out.clamp(1e-5, 1. - 1e-5)
            out = logistic.mixture_inv_cdf(out, pi, mu, s)
            logistic_ldj = logistic.mixture_log_pdf(out, pi, mu, s)
            j = - (a + scale_ldj + logistic_ldj).flatten(1).sum(-1)
        else:
            out = logistic.mixture_log_cdf(x_change, pi, mu, s).exp()
            out, scale_ldj = logistic.inverse(out)
            out = (out + b) * a.exp()
            logistic_ldj = logistic.mixture_log_pdf(x_change, pi, mu, s)
            j = (logistic_ldj + scale_ldj + a).flatten(1).sum(-1)

        x = torch.cat((x_id, out), dim=1)
        return (x,), j