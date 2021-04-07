import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from FrEIA.modules import InvertibleModule 

class InvertibleConv1x1(InvertibleModule):
    def __init__(self, dims_in, dims_c=None, LU_decomposed=False):
        super().__init__(dims_in, dims_c)
        num_channels = dims_in[0][0]
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        self.w_shape = w_shape
        self.LU = LU_decomposed

    def get_weight(self, input, reverse):
        w_shape = self.w_shape
        #pixels = thops.pixels(input)
        dlogdet = torch.slogdet(self.weight)[1] * int(input.size(2) * input.size(3))
        dlogdet = dlogdet.repeat((input.size(0))) #new addition to take into account the batch dimension.
        if not reverse:
            weight = self.weight.view(w_shape[0], w_shape[1], 1, 1)
        else:
            weight = torch.inverse(self.weight.double()).float() \
                .view(w_shape[0], w_shape[1], 1, 1)
        return weight, dlogdet

    def forward(self, x, c=None, jac=True, rev=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        input=x[0]
        weight, dlogdet = self.get_weight(input, rev)
        
        if not rev:
            z = F.conv2d(input, weight)
            return (z,), dlogdet
        else:
            z = F.conv2d(input, weight)
            return (z,), dlogdet