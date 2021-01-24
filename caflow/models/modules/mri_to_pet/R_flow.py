#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 22:02:53 2021

@author: gbatz97
"""




from caflow.models.modules.blocks.AffineCouplingLayer import AffineCouplingLayer
import torch.nn as nn
from iunets.layers import InvertibleDownsampling1D, InvertibleDownsampling2D, InvertibleDownsampling3D, \
                          InvertibleChannelMixing1D, InvertibleChannelMixing2D, InvertibleChannelMixing3D
import torch


class flow_block(nn.Module):
    def __init__(self, channels, dim, depth, network):
        super(flow_block, self).__init__()
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
                                                   dim=dim, network=network, 
                                                   mask_info={'mask_type':'channel', 'invert':False}))
    
    def forward(self, h, logdet, reverse):
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
        for layer in self.layers:
            if isinstance(layer, self.InvertibleDownsampling) or isinstance(layer, self.InvertibleChannelMixing):
                h = layer.inverse(h) #we are following the implementational change of InvertibleDownsampling and InvertibleChannelMixing
            else:
                h, logdet = layer(h, logdet, reverse=True)
        
        return h, logdet


class R_flow(nn.Module):
    def __init__(self, channels, dim, scales, scale_depth, network, opts):
        super(R_flow, self).__init__()
        
        self.channels = channels
        self.dim = dim
        self.scales = scales
        
        self.scale_blocks = nn.ModuleList()
        
        for scale in range(self.scales):
            scale_channels = 2 ** (dim * scale) * self.channels
            self.scale_blocks.append(flow_block(channels = scale_channels, 
                                                depth = scale_depth, 
                                                network = network))

        # Create prior distribution for final latent space
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)
        self.device = opts.device['R_flow']

    def forward(self, y=None, z=None, logprior=0., logdet=0., reverse=False):

        if reverse:
            assert z is not None
            y_dec, logdet = self.decode(z, logdet=logdet)
            return y_dec, logdet
        else:
            assert y is not None
            z_enc, logprior, logdet = self.encode(y, logprior=logprior, logdet=logdet)
            return z_enc, logprior, logdet
    
    def encode(self, y, logprior, logdet):
        #y is the HR image/scan that we want to encode in the latent space
        #z_enc: list of the encoded latent tensors in ascending scale order
        #order: from scale 1 (dim=orig_dim/2) --- to --- scale n (dim=orig_dim/2^n)

        h_pass = y
        z_enc = []
        for i in range(self.scales):
            if i==self.scales-1:
                h_split, logdet = self.scale_blocks[i](h=h_pass, logdet=logdet, reverse=False)
            else:
                h, logdet = self.scale_blocks[i](h=h_pass, logdet=logdet, reverse=False)
                h_split, h_pass = h.chunk(2, dim=1)
            
            logprior+=self.prior.log_prob(h_split).sum()
            z_enc.append(h_split)

        return z_enc, logprior, logdet


    def decode(self, z, logdet):
        #z is a list of the latent tensors of the different scales.
        #The tensors of different scales have been put in an ascending order
        #z = [h_split(1st scale)-size:D/2, ..., h_split(nth scale)-size:D/2^n]

        h_pass=None
        for i in range(self.scales):
            h_split = z[self.scales-1-i]
            if h_pass==None:
                concat_pass = h_split
            else:
                concat_pass = torch.cat([h_split, h_pass], dim=1)
            h_pass, logdet = self.scale_blocks[i](h=concat_pass, logdet=logdet, reverse=True)

        return h_pass, logdet


        

        

