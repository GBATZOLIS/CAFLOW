from iunets.layers import InvertibleChannelMixing1D, InvertibleChannelMixing2D, InvertibleChannelMixing3D
import torch.nn as nn

class ChannelMixingLayer(nn.Module):
    def __init__(self, channels, dim, method='cayley', learnable=True):
        super(ChannelMixingLayer, self).__init__()
        self.i_channelmixinglayer = [InvertibleChannelMixing1D, InvertibleChannelMixing2D, InvertibleChannelMixing3D][dim-1](channels, method, learnable)
    
    def forward(self, x, rev=False):
        if not rev:
            x = self.i_channelmixinglayer(x[0])
        else:
            x = self.i_channelmixinglayer.inverse(x[0])
        return (x,), 0.