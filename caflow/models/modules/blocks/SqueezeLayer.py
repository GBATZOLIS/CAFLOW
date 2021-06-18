from iunets.layers import InvertibleDownsampling1D, InvertibleDownsampling2D, InvertibleDownsampling3D
import torch.nn as nn

class SqueezeLayer(nn.Module):
    def __init__(self, channels, dim, stride=2, method='cayley', init='squeeze', learnable=False):
        super(SqueezeLayer, self).__init__()
        self.i_squeezelayer = [InvertibleDownsampling1D, InvertibleDownsampling2D, InvertibleDownsampling3D][dim-1](channels, stride, method, init, learnable)
    
    def forward(self, x, rev=False):
        if not rev:
            x = self.i_squeezelayer(x)
        else:
            x = self.i_squeezelayer.inverse(x)
        return x, 0.