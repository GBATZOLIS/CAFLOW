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
from caflow.models.modules.networks.GatedConvNet import GatedConvNet



class FlowBlock(nn.Module):
    def __init__(self, channels, dim, depth):
        super(FlowBlock, self).__init__()
        #shape: (channels, X, Y, Z) for 3D, (channels, X, Y) for 2D
        #we intend to use fully convolutional models which means that we do not need the real shape. We just need the input channels
        self.channels = channels 
        self.dim = dim
        self.depth = depth

        self.layers = nn.ModuleList()
        
        #choose the correct invertible downsampling and channel mixining layers from the iUNETS library based on the dimension of the tensor
        self.InvertibleDownsampling = [InvertibleDownsampling1D, InvertibleDownsampling2D, InvertibleDownsampling3D][dim-1]
        self.InvertibleChannelMixing = [InvertibleChannelMixing1D, InvertibleChannelMixing2D, InvertibleChannelMixing3D][dim-1]
        
        self.layers.append(self.InvertibleDownsampling(in_channels = channels, stride=2, method='cayley', init='squeeze', learnable=True))
        #new shape: 3D -> (8*channels, X/2, Y/2, Z/2)
        #           2D -> (4*channels, X/2, Y/2)
        #           1D -> (2*channels, X/2)
        
        transformed_channels = 2**dim*channels

        for _ in range(depth):
            #append activation layer
            #append permutation layer
            #append the affine coupling layer

            self.layers.append(self.InvertibleChannelMixing(in_channels = transformed_channels, 
                                                            method = 'cayley', learnable=True))

            self.layers.append(AffineCouplingLayer(c_in = transformed_channels, 
                                                   dim=dim, mask_info={'mask_type':'channel', 'invert':False},
                                                   network=GatedConvNet(c_in=transformed_channels, dim=dim, 
                                                                   c_hidden=2*transformed_channels, 
                                                                   c_out=-1, num_layers=1)))
    
    def forward(self, h, logdet, reverse=False):
        if reverse:
            h_pass, logdet = self.decode(h, logdet)
        else:
            h_pass, logdet = self.encode(h, logdet)
        
        return h_pass, logdet
    
    def encode(self, h, logdet):
        for layer in self.layers:
            if isinstance(layer, self.InvertibleDownsampling) or isinstance(layer, self.InvertibleChannelMixing):
                #The InvertibleDownsampling and InvertibleChannelMixing Layers introduced by Christian et al. yield unit determinant
                #This is why they do not contribute to the logdet summation.
                h = layer(h)
            else:
                h, logdet = layer(h, logdet, reverse=False)
        
        return h, logdet

    def decode(self, h, logdet):
        for layer in reversed(self.layers):
            if isinstance(layer, self.InvertibleDownsampling) or isinstance(layer, self.InvertibleChannelMixing):
                h = layer.inverse(h) #we are following the implementational change of InvertibleDownsampling and InvertibleChannelMixing
            else:
                h, logdet = layer(h, logdet, reverse=True)
        
        return h, logdet