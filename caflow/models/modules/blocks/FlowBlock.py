#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 17:24:21 2021

@author: gbatz97
"""

import torch.nn as nn
from caflow.models.modules.blocks.ActNorm import ActNorm
from caflow.models.modules.blocks.permutations import InvertibleConv1x1
from caflow.models.modules.blocks import coupling_layer

from caflow.models.modules.networks.GatedConvNet import GatedConvNet
from caflow.models.modules.networks.SimpleConvNet import SimpleConvNet
from caflow.models.modules.networks.nnflowpp import nnflowpp
from caflow.models.modules.networks.parse_nn_by_name import parse_nn_by_name
from caflow.models.modules.networks.CondSimpleConvNet import CondSimpleConvNet
import FrEIA.modules as Fm
from caflow.models.modules.blocks.AffineCouplingLayer import AffineCouplingOneSided

class FlowBlock(nn.Module):
    def __init__(self, channels, dim, resolution, depth, coupling_type, nn_settings):
        super(FlowBlock, self).__init__()
        #shape: (channels, X, Y, Z) for 3D, (channels, X, Y) for 2D
        #we intend to use fully convolutional models which means that we do not need the real shape. We just need the input channels
        self.channels = channels 
        self.dim = dim
        self.resolution = resolution
        self.depth = depth

        self.layers = nn.ModuleList()

        self.layers.append(Fm.IRevNetDownsampling(dims_in=[(channels,)+resolution]))
        #new shape: 3D -> (8*channels, X/2, Y/2, Z/2)
        #           2D -> (4*channels, X/2, Y/2)
        #           1D -> (2*channels, X/2)
        
        transformed_channels = 2**dim*channels
        transformed_resolution = tuple([x//2 for x in self.resolution])
        dims_in = [(transformed_channels,)+transformed_resolution]

        #transition step
        for _ in range(2):
            self.layers.append(Fm.ActNorm(dims_in=dims_in))
            self.layers.append(InvertibleConv1x1(dims_in=dims_in))

        for _ in range(depth):
            self.layers.append(Fm.ActNorm(dims_in=dims_in))
            self.layers.append(InvertibleConv1x1(dims_in=dims_in))
            self.layers.append(coupling_layer(coupling_type)(dims_in=dims_in, \
                                                subnet_constructor=parse_nn_by_name(nn_settings['nn_type']),
                                                nn_settings=nn_settings))
    
    def forward(self, h, logdet, reverse=False):
        if reverse:
            h_pass, logdet = self.decode(h, logdet)
        else:
            h_pass, logdet = self.encode(h, logdet)
        return h_pass, logdet
    
    def encode(self, h, logdet):
        h=(h,)
        for layer in self.layers:
            h, jac = layer(h, rev=False)
            logdet += jac
        return h[0], logdet

    def decode(self, h, logdet):
        h=(h,)
        for layer in reversed(self.layers):
            h, jac = layer(h, rev=True)
            logdet += jac
        return h[0], logdet