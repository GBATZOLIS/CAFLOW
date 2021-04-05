from FrEIA.modules import InvertibleModule

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import conv2d, conv_transpose2d


class ActNorm(InvertibleModule):
    def __init__(self, dims_in, dims_c=None, init_data=None):
        super().__init__(dims_in, dims_c)
        self.dims_in = dims_in[0]
        param_dims = [1, self.dims_in[0]] + [1 for i in range(len(self.dims_in) - 1)]
        self.scale = nn.Parameter(torch.zeros(*param_dims))
        self.bias = nn.Parameter(torch.zeros(*param_dims))

        if init_data:
            self.initialize_with_data(init_data)
        else:
            self.init_on_next_batch = True

        def on_load_state_dict(*args):
            # when this module is loading state dict, we SHOULDN'T init with data,
            # because that will reset the trained parameters. Registering a hook
            # that disable this initialisation.
            self.init_on_next_batch = False
        self._register_load_state_dict_pre_hook(on_load_state_dict)


    def initialize_with_data(self, data):
        # Initialize to mean 0 and std 1 with sample batch
        # 'data' expected to be of shape (batch, channels[, ...])
        assert all([data.shape[i+1] == self.dims_in[i] for i in range(len(self.dims_in))]),\
            "Can't initialize ActNorm layer, provided data don't match input dimensions."
        self.scale.data.view(-1)[:] \
            = torch.log(1 / data.transpose(0,1).contiguous().view(self.dims_in[0], -1).std(dim=-1))
        data = data * self.scale.exp()
        self.bias.data.view(-1)[:] \
            = -data.transpose(0,1).contiguous().view(self.dims_in[0], -1).mean(dim=-1)
        self.init_on_next_batch = False


    def forward(self, x, rev=False, jac=True):
        if self.init_on_next_batch:
            self.initialize_with_data(x[0])

        jac = (self.scale.sum() * np.prod(self.dims_in[1:])).repeat(x[0].shape[0])
        if rev:
            jac = -jac

        if not rev:
            return [x[0] * self.scale.exp() + self.bias], jac
        else:
            return [(x[0] - self.bias) / self.scale.exp()], jac


    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        return input_dims