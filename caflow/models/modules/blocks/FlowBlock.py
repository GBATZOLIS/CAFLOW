#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 17:24:21 2021

@author: gbatz97
"""

import torch.nn as nn
from iunets.iunets.layers import InvertibleDownsampling1D, InvertibleDownsampling2D, InvertibleDownsampling3D, \
                          InvertibleChannelMixing1D, InvertibleChannelMixing2D, InvertibleChannelMixing3D
from caflow.models.modules.blocks.AffineCouplingLayer import AffineCouplingLayer
from caflow.models.modules.blocks.ActNorm import ActNorm
from caflow.models.modules.blocks.permutations import InvertibleConv1x1

from caflow.models.modules.networks.GatedConvNet import GatedConvNet
from caflow.models.modules.networks.SimpleConvNet import SimpleConvNet
from caflow.models.modules.networks.CondSimpleConvNet import CondSimpleConvNet
import FrEIA.modules as Fm

class FlowBlock(nn.Module):
    def __init__(self, channels, dim, resolution, depth):
        super(FlowBlock, self).__init__()
        #shape: (channels, X, Y, Z) for 3D, (channels, X, Y) for 2D
        #we intend to use fully convolutional models which means that we do not need the real shape. We just need the input channels
        self.channels = channels 
        self.dim = dim
        self.resolution = resolution
        self.depth = depth

        self.layers = nn.ModuleList()
        
        #choose the correct invertible downsampling and channel mixining layers from the iUNETS library based on the dimension of the tensor
        self.InvertibleDownsampling = [InvertibleDownsampling1D, InvertibleDownsampling2D, InvertibleDownsampling3D][dim-1]
        #self.InvertibleChannelMixing = [InvertibleChannelMixing1D, InvertibleChannelMixing2D, InvertibleChannelMixing3D][dim-1]
        #self.layers.append(self.InvertibleDownsampling(in_channels = channels, stride=2, method='cayley', init='squeeze', learnable=False))
        self.layers.append(Fm.IRevNetDownsampling(dims_in=[(channels,)+resolution]))
        #new shape: 3D -> (8*channels, X/2, Y/2, Z/2)
        #           2D -> (4*channels, X/2, Y/2)
        #           1D -> (2*channels, X/2)
        
        transformed_channels = 2**dim*channels
        transformed_resolution = tuple([x//2 for x in self.resolution])

        #transition step
        #for _ in range(2):
        #    self.layers.append(ActNorm(num_features=transformed_channels, dim=dim))
        #    self.layers.append(InvertibleConv1x1(num_channels = transformed_channels))

        dims_in = [(transformed_channels,)+transformed_resolution]
        for i in range(depth):
            self.layers.append(Fm.AllInOneBlock(dims_in=dims_in, \
                                                subnet_constructor=SimpleConvNet))
    
    def forward(self, h, logdet, reverse=False):
        if reverse:
            h_pass, logdet = self.decode(h, logdet)
        else:
            h_pass, logdet = self.encode(h, logdet)
        return h_pass, logdet
    
    def encode(self, h, logdet):
        h=(h,)
        for layer in self.layers:
            if isinstance(layer, self.InvertibleDownsampling):
                h = layer(h[0])
                h = (h,)
            else:
                h, jac = layer(h, rev=False)
                logdet += jac
        return h[0], logdet

    def decode(self, h, logdet):
        h=(h,)
        for layer in reversed(self.layers):
            if isinstance(layer, self.InvertibleDownsampling):
                h = layer.inverse(h[0])
                h = (h,)
            else:
                h, jac = layer(h, rev=True)
                logdet += jac
        return h[0], logdet