#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 01:31:06 2021

@author: gbatz97
"""


import torch.nn as nn

from iunets.iunets.layers import InvertibleDownsampling1D, InvertibleDownsampling2D, InvertibleDownsampling3D, \
                                 InvertibleChannelMixing1D, InvertibleChannelMixing2D, InvertibleChannelMixing3D
                          
from caflow.models.modules.blocks.AffineCouplingLayer import AffineCouplingLayer
from caflow.models.modules.blocks.AffineInjector import AffineInjector
from caflow.models.modules.blocks.ActNorm import ActNorm

from caflow.models.modules.networks.CondGatedConvNet import CondGatedConvNet
from caflow.models.modules.networks.SimpleConvNet import SimpleConvNet
from caflow.models.modules.networks.CondSimpleConvNet import CondSimpleConvNet

class g_S(nn.Module):
    def __init__(self, channels, dim, depth, last_scale):
        super(g_S, self).__init__()
        #shape: (channels, X, Y, Z) for 3D, (channels, X, Y) for 2D
        #we intend to use fully convolutional models which means that we do not need the real shape. We just need the input channels
        self.channels = channels 
        self.dim = dim
        self.depth = depth

        self.layers = nn.ModuleList()
        
        #choose the correct invertible downsampling and channel mixining layers from the iUNETS library based on the dimension of the tensor
        self.InvertibleDownsampling = [InvertibleDownsampling1D, InvertibleDownsampling2D, InvertibleDownsampling3D][dim-1]
        self.InvertibleChannelMixing = [InvertibleChannelMixing1D, InvertibleChannelMixing2D, InvertibleChannelMixing3D][dim-1]
        #print(channels)
        self.layers.append(self.InvertibleDownsampling(in_channels = channels, stride=2, method='cayley', init='squeeze', learnable=True))
        #new shape: 3D -> (8*channels, X/2, Y/2, Z/2)
        #           2D -> (4*channels, X/2, Y/2)
        #           1D -> (2*channels, X/2)
        
        transformed_channels = 2**dim*channels

        for _ in range(depth):
            #append activation layer
            self.layers.append(ActNorm(num_features=transformed_channels, dim=dim))

            #append permutation layer
            self.layers.append(self.InvertibleChannelMixing(in_channels = transformed_channels, 
                                                            method = 'cayley', learnable=True))
            
            #AFFINE INJECTOR
            #self.layers.append(AffineInjector(c_in=transformed_channels, dim=dim, 
            #                                  network = CondSimpleConvNet(c_in = transformed_channels, dim=dim,
            #                                                    c_hidden = 2*transformed_channels, c_out=-1, num_layers=1,
            #                                                    layer_type='injector', num_cond_rvs=2, last_scale=last_scale)))
            
            #AFFINE COUPLING LAYER
            self.layers.append(AffineCouplingLayer(c_in = transformed_channels, dim=dim, 
                                                   mask_info={'mask_type':'channel', 'invert':False},
                                                   network = CondSimpleConvNet(c_in = transformed_channels, dim=dim,
                                                                     c_hidden = 3*transformed_channels, c_out=-1, num_layers=1,
                                                                     layer_type='coupling', num_cond_rvs=2, last_scale=last_scale)))
            
    
    def forward(self, h, L, D, logdet, reverse=False):
        if reverse:
            h_pass, logdet = self.decode(h, L, D, logdet)
        else:
            h_pass, logdet = self.encode(h, L, D, logdet)
        
        return h_pass, logdet
    
    def encode(self, h, L, D, logdet):
        for layer in self.layers:
            if isinstance(layer, self.InvertibleDownsampling) or isinstance(layer, self.InvertibleChannelMixing):
                #The InvertibleDownsampling and InvertibleChannelMixing Layers introduced by Christian et al. yield unit determinant
                #This is why they do not contribute to the logdet summation.
                h = layer(h)
            elif isinstance(layer, ActNorm):
                h, logdet = layer(h, logdet, reverse=False)
            else:
                h, logdet = layer(h, logdet, reverse=False, cond_rv=[L, D])
        
        return h, logdet

    def decode(self, h, L, D, logdet):
        for layer in reversed(self.layers):
            if isinstance(layer, self.InvertibleDownsampling) or isinstance(layer, self.InvertibleChannelMixing):
                h = layer.inverse(h) #we are following the implementational change of InvertibleDownsampling and InvertibleChannelMixing
            elif isinstance(layer, ActNorm):
                h, logdet = layer(h, logdet, reverse=True)
            else:
                h, logdet = layer(h, logdet, reverse=True, cond_rv=[L, D])
        
        return h, logdet




class g_I(nn.Module):
    def __init__(self, channels, dim, depth):
        super(g_I, self).__init__()
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
            self.layers.append(ActNorm(num_features=transformed_channels, dim=dim))

            #append permutation layer
            self.layers.append(self.InvertibleChannelMixing(in_channels = transformed_channels, 
                                                            method = 'cayley', learnable=True))

            #AFFINE INJECTOR
            #self.layers.append(AffineInjector(c_in=transformed_channels, dim=dim, 
            #                                  network = CondSimpleConvNet(c_in = transformed_channels, dim=dim,
            #                                                    c_hidden = 2*transformed_channels, c_out=-1, num_layers=1,
            #                                                    layer_type='injector', num_cond_rvs=1)))
            
            #AFFINE COUPLING LAYER
            self.layers.append(AffineCouplingLayer(c_in = transformed_channels, dim=dim, 
                                                   mask_info={'mask_type':'channel', 'invert':False},
                                                   network = CondSimpleConvNet(c_in = transformed_channels, dim=dim,
                                                                     c_hidden = 3*transformed_channels, c_out=-1, num_layers=1,
                                                                     layer_type='coupling', num_cond_rvs=1)))
            
    
    def forward(self, h, D, logdet, reverse=False):
        if reverse:
            h_pass, logdet = self.decode(h, D, logdet)
        else:
            h_pass, logdet = self.encode(h, D, logdet)
        
        return h_pass, logdet
    
    def encode(self, h, D, logdet):
        for layer in self.layers:
            if isinstance(layer, self.InvertibleDownsampling) or isinstance(layer, self.InvertibleChannelMixing):
                #The InvertibleDownsampling and InvertibleChannelMixing Layers introduced by Christian et al. yield unit determinant
                #This is why they do not contribute to the logdet summation.
                h = layer(h)
            elif isinstance(layer, ActNorm):
                h, logdet = layer(h, logdet, reverse=False)
            else:
                h, logdet = layer(h, logdet, reverse=False, cond_rv=[D])
        
        return h, logdet

    def decode(self, h, D, logdet):
        for layer in reversed(self.layers):
            if isinstance(layer, self.InvertibleDownsampling) or isinstance(layer, self.InvertibleChannelMixing):
                h = layer.inverse(h) #we are following the implementational change of InvertibleDownsampling and InvertibleChannelMixing
            elif isinstance(layer, ActNorm):
                h, logdet = layer(h, logdet, reverse=True)
            else:
                h, logdet = layer(h, logdet, reverse=True, cond_rv=[D])
        
        return h, logdet