#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 22:02:53 2021

@author: gbatz97
"""

import torch.nn as nn
import torch
from caflow.models.modules.blocks.FlowBlock import FlowBlock                    

class UnconditionalFlow(nn.Module):
    def __init__(self, channels, dim, scales, scale_depth):
        super(UnconditionalFlow, self).__init__()
        
        self.channels = channels
        self.dim = dim
        self.scales = scales
        
        self.scale_blocks = nn.ModuleList()
        
        for scale in range(self.scales):
            scale_channels = self.calculate_scale_channels(dim, scale)
            self.scale_blocks.append(FlowBlock(channels = scale_channels,
                                               dim = dim,
                                               depth = scale_depth))

        # Create prior distribution for final latent space
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)
    
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
        for i in range(self.scales):
            if i==self.scales-1:
                h_split, logdet = self.scale_blocks[i](h=h_pass, logdet=logdet, reverse=False)
            else:
                h, logdet = self.scale_blocks[i](h=h_pass, logdet=logdet, reverse=False)
                h_split, h_pass = h.chunk(2, dim=1)
            
            logprior+=self.prior.log_prob(h_split).sum(dim = [i+1 for i in range(self.dim+1)])
            z_enc.append(h_split)

        return z_enc, logprior, logdet


    def decode(self, z:list, logdet):
        #z is a list of the latent tensors of the different scales.
        #The tensors of different scales have been put in an ascending order
        #z = [h_split(1st scale)-size:D/2, ..., h_split(nth scale)-size:D/2^n]
        print(len(z))
        
        h_pass=None
        for i in range(self.scales):
            h_split = z[self.scales-1-i]
            if h_pass==None:
                concat_pass = h_split
            else:
                concat_pass = torch.cat([h_split, h_pass], dim=1)
            h_pass, logdet = self.scale_blocks[self.scales-1-i](h=concat_pass, logdet=logdet, reverse=True)

        return h_pass, logdet



rflow = UnconditionalFlow(channels=1, dim=3, scales=4, scale_depth=3)
y = torch.randn((4,1,128,128,128), dtype=torch.float32)
z_enc, logprior, logdet = rflow(y=y)

print(len(z_enc))
for elem in z_enc:
    print(elem.size())
y_dec, logdet = rflow(z=z_enc, reverse=True)

r = y-y_dec
print(r.sum())

        

