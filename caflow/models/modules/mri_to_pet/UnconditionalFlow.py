#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 22:02:53 2021

@author: gbatz97
"""

import torch.nn as nn
import torch
from caflow.models.modules.blocks.FlowBlock import FlowBlock                    
from caflow.models.modules.blocks.Dequantisation import Dequantisation, VariationalDequantization
from caflow.models.modules.blocks.InvertibleScaling import InvertibleScaling
class UnconditionalFlow(nn.Module):
    def __init__(self, channels, dim, resolution, scales, scale_depth, use_inv_scaling, scaling_range, use_dequantisation, quants, vardeq_depth, coupling_type, nn_settings):
        super(UnconditionalFlow, self).__init__()
        
        self.channels = channels
        self.dim = dim
        self.resolution = resolution
        self.scales = scales
        self.use_dequantisation = use_dequantisation
        self.use_inv_scaling = use_inv_scaling

        self.scale_blocks = nn.ModuleList()

        if self.use_dequantisation:
            if vardeq_depth is None:
                self.dequantisation = Dequantisation(dim=dim, quants=quants)
            else:
                self.dequantisation = VariationalDequantization(channels=channels, depth=vardeq_depth, dim=dim, \
                                                            resolution=self.calculate_resolution(dim, 0),\
                                                            quants=quants, coupling_type=coupling_type, nn_settings=nn_settings)

        if self.use_inv_scaling:
            self.invertible_scaling = InvertibleScaling(scaling_range)

        for scale in range(self.scales):
            scale_channels = self.calculate_scale_channels(dim, scale)
            resolution = self.calculate_resolution(dim, scale)
            self.scale_blocks.append(FlowBlock(channels = scale_channels, dim = dim,
                                               resolution=resolution, depth = scale_depth,
                                               coupling_type=coupling_type, nn_settings=nn_settings))

        # Create prior distribution for final latent space
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)
    
    def calculate_resolution(self, dim, scale):
        if isinstance(self.resolution, int):
            resolution = tuple([self.resolution//2**scale for _ in range(self.dim)])
        else:
            resolution = tuple([x//2**scale for x in self.resolution])
        return resolution

    def calculate_scale_channels(self, dim, scale):
        if scale==0:
            return  2 ** (dim * scale) * self.channels
        else:
            return 2 ** ((dim-1) * scale) * self.channels
        
        
    def forward(self, y=None, z=[], logprior=0., logdet=0., reverse=False):
        if reverse:
            #assert z is not None
            y_dec, logdet = self.decode(z, logdet=logdet)
            return y_dec, logdet
        else:
            #assert y is not None
            z_enc, logprior, logdet = self.encode(y, logprior=logprior, logdet=logdet)
            return z_enc, logprior, logdet
    
    def encode(self, y, logprior, logdet):
        #y is the HR image/scan that we want to encode in the latent space
        #z_enc: list of the encoded latent tensors in ascending scale order
        #order: from scale 1 (dim=orig_dim/2) --- to --- scale n (dim=orig_dim/2^n)

        h_pass = y
        z_enc = []
        
        if self.use_dequantisation and not self.use_inv_scaling:
            h_pass, logdet = self.dequantisation(h_pass, logdet, False) #dequantisation
        elif not self.use_dequantisation and self.use_inv_scaling:
            h_pass, logdet = self.invertible_scaling(h_pass, logdet, False)
        elif self.use_dequantisation and self.use_inv_scaling:
            raise Exception('Dequantisation and Invertible scaling should not be used together. ')

        for i in range(0, self.scales):
            if i==self.scales-1:
                h_split, logdet = self.scale_blocks[i](h_pass, logdet, False)
            else:
                h, logdet = self.scale_blocks[i](h_pass, logdet, False)
                h_split, h_pass = h.chunk(2, dim=1)
            
            logprior+=self.prior.log_prob(h_split).sum(dim = [i+1 for i in range(self.dim+1)])
            z_enc.append(h_split)

        return z_enc, logprior, logdet


    def decode(self, z:list, logdet):
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
            h_pass, logdet = self.scale_blocks[self.scales-i-1](concat_pass, logdet, True)
        
        if self.use_dequantisation and not self.use_inv_scaling:
            h_pass, logdet = self.dequantisation(h_pass, logdet, True) #quantisation
        elif not self.use_dequantisation and self.use_inv_scaling:
            h_pass, logdet = self.invertible_scaling(h_pass, logdet, True)
        elif self.use_dequantisation and self.use_inv_scaling:
            raise Exception('Dequantisation and Invertible scaling should not be used together. ')

        return h_pass, logdet


"""
#instantiate the unconditional flow
rflow = UnconditionalFlow(channels=1, dim=3, scales=4, scale_depth=3, network = GatedConvNet)

y = torch.randn((2, 1, 64, 64, 64), dtype=torch.float32)
print('y shape: ', y.size())

print('Encoding y with the forward pass...We get z_enc (same dimensionality)')
z_enc, logprior, logdet = rflow(y=y)

print('z_enc elements:')
for i, elem in enumerate(z_enc):
    print(i, elem.size())
    
print('logprior size: ', logprior.size())
print('logdet size: ', logdet.size())

print('Decoding y_dec from its z_enc enconding... We pass z_enc through the backward pass.')
y_dec, logdet = rflow(z=z_enc, reverse=True)
print('y_dec size:', y_dec.size())

r = torch.abs(y-y_dec)
print('sum(|y-y_dec|)',torch.sum(r))
print('mean(|y-y_dec|):',torch.mean(r))
"""
        

