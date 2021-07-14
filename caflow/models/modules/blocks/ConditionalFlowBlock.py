#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 01:31:06 2021

@author: gbatz97
"""

import torch.nn as nn
import torch
from caflow.models.modules.networks.parse_nn_by_name import parse_nn_by_name
from caflow.models.modules.blocks.permutations import InvertibleConv1x1
from caflow.models.modules.blocks.AffineCouplingLayer import AffineCouplingOneSided, ConditionalAffineTransform, AdditiveCouplingOneSided, ConditionalAdditiveTransform
import FrEIA.modules as Fm
from caflow.utils.processing import squeeze, general_squeeze
from caflow.models.modules.blocks.SqueezeLayer import SqueezeLayer
from caflow.models.modules.blocks.ChannelMixingLayer import ChannelMixingLayer

class g_S(nn.Module):
    def __init__(self, channels, dim, resolution, depth, nn_settings, last_scale):
        super(g_S, self).__init__()
        #shape: (channels, X, Y, Z) for 3D, (channels, X, Y) for 2D
        #we intend to use fully convolutional models which means that we do not need the real shape. We just need the input channels
        self.channels = channels 
        self.dim = dim
        self.resolution = resolution
        self.depth = depth

        self.layers = nn.ModuleList()
        
        self.layers.append(SqueezeLayer(channels, dim))
        #new shape: 3D -> (8*channels, X/2, Y/2, Z/2)
        #           2D -> (4*channels, X/2, Y/2)
        #           1D -> (2*channels, X/2)
        
        transformed_channels = 2**dim*channels
        transformed_resolution = tuple([x//2 for x in self.resolution])
        dims_in = [(transformed_channels,)+transformed_resolution]

        if last_scale:
            conditional_dims = (2*transformed_channels,)+transformed_resolution
        else:
            conditional_dims = dims_in[0]  
        dims_c = [conditional_dims, conditional_dims]

        #transition step
        #for _ in range(1):
        #    self.layers.append(Fm.ActNorm(dims_in=dims_in))
        #    self.layers.append(InvertibleConv1x1(dims_in=dims_in))

        for _ in range(depth):
            #append activation layer
            self.layers.append(Fm.ActNorm(dims_in=dims_in))

            #append permutation layer
            #self.layers.append(InvertibleConv1x1(dims_in=dims_in))
            self.layers.append(ChannelMixingLayer(transformed_channels, dim))
            
            #AFFINE INJECTOR
            self.layers.append(ConditionalAdditiveTransform(dims_in=dims_in, \
                                                dims_c=dims_c, 
                                                subnet_constructor=parse_nn_by_name(nn_settings['nn_type']),
                                                clamp=1, clamp_activation = (lambda u: 0.5*torch.sigmoid(u)+0.5),
                                                nn_settings=nn_settings))
            
            #AFFINE COUPLING LAYER
            self.layers.append(AdditiveCouplingOneSided(dims_in=dims_in, 
                                                      dims_c=dims_c,
                                                      subnet_constructor=parse_nn_by_name(nn_settings['nn_type']),
                                                      clamp=1, clamp_activation = (lambda u: 0.5*torch.sigmoid(u)+0.5),
                                                      nn_settings=nn_settings))
    
    def forward(self, h, L, D, logdet, reverse=False):
        if reverse:
            h_pass, logdet = self.decode(h, L, D, logdet)
        else:
            h_pass, logdet = self.encode(h, L, D, logdet)
        
        return h_pass, logdet
    
    def encode(self, h, L, D, logdet):
        h = (h,)
        for layer in self.layers:
            if isinstance(layer, (AffineCouplingOneSided, ConditionalAffineTransform)):
                h, jac = layer(h, c=[general_squeeze(L), general_squeeze(D)], rev=False)
            else:
                h, jac = layer(h, rev=False)

            logdet += jac

        return h[0], logdet

    def decode(self, h, L, D, logdet):
        h = (h,)
        for layer in reversed(self.layers):
            if isinstance(layer, (AffineCouplingOneSided, ConditionalAffineTransform)):
                h, jac = layer(h, c=[general_squeeze(L), general_squeeze(D)], rev=True)
            else:
                h, jac = layer(h, rev=True)

            logdet += jac
        
        return h[0], logdet



class g_I(nn.Module):
    def __init__(self, channels, dim, resolution, depth, nn_settings):
        super(g_I, self).__init__()
        #shape: (channels, X, Y, Z) for 3D, (channels, X, Y) for 2D
        #we intend to use fully convolutional models which means that we do not need the real shape. We just need the input channels
        self.channels = channels 
        self.dim = dim
        self.resolution = resolution
        self.depth = depth

        self.layers = nn.ModuleList()
        
        self.layers.append(SqueezeLayer(channels, dim))
        #new shape: 3D -> (8*channels, X/2, Y/2, Z/2)
        #           2D -> (4*channels, X/2, Y/2)
        #           1D -> (2*channels, X/2)
        
        transformed_channels = 2**dim*channels
        transformed_resolution = tuple([x//2 for x in self.resolution])
        dims_in = [(transformed_channels,)+transformed_resolution]
        dims_c = dims_in

        #transition step
        #for _ in range(1):
        #    self.layers.append(Fm.ActNorm(dims_in=dims_in))
        #    self.layers.append(InvertibleConv1x1(dims_in=dims_in))

        for _ in range(depth):
            #append activation layer
            self.layers.append(Fm.ActNorm(dims_in=dims_in))

            #append permutation layer
            #self.layers.append(InvertibleConv1x1(dims_in=dims_in))
            self.layers.append(ChannelMixingLayer(transformed_channels, dim))
            
            #AFFINE INJECTOR
            self.layers.append(ConditionalAdditiveTransform(dims_in=dims_in, \
                                                dims_c=dims_c, 
                                                subnet_constructor=parse_nn_by_name(nn_settings['nn_type']),
                                                clamp=1, clamp_activation = (lambda u: 0.5*torch.sigmoid(u)+0.5),
                                                nn_settings=nn_settings))
            
            #AFFINE COUPLING LAYER
            self.layers.append(AdditiveCouplingOneSided(dims_in=dims_in, 
                                                      dims_c=dims_c,
                                                      subnet_constructor=parse_nn_by_name(nn_settings['nn_type']), 
                                                      clamp=1, clamp_activation = (lambda u: 0.5*torch.sigmoid(u)+0.5),
                                                      nn_settings=nn_settings))
    
    def forward(self, h, D, logdet, reverse=False):
        if reverse:
            h_pass, logdet = self.decode(h, D, logdet)
        else:
            h_pass, logdet = self.encode(h, D, logdet)
        
        return h_pass, logdet
    
    def encode(self, h, D, logdet):
        h = (h,)
        for layer in self.layers:
            if isinstance(layer, (AffineCouplingOneSided, ConditionalAffineTransform)):
                h, jac = layer(h, c=[general_squeeze(D)], rev=False)
            else:
                h, jac = layer(h, rev=False)

            logdet += jac

        return h[0], logdet

    def decode(self, h, D, logdet):
        h = (h,)
        for layer in reversed(self.layers):
            if isinstance(layer, (AffineCouplingOneSided, ConditionalAffineTransform)):
                h, jac = layer(h, c=[general_squeeze(D)], rev=True)
            else:
                h, jac = layer(h, rev=True)

            logdet += jac
        
        return h[0], logdet